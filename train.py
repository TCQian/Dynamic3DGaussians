import copy
import json
import os
from argparse import ArgumentParser
from random import randint

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from PIL import Image
from tqdm import tqdm

from dnerf import get_dataset_dnerf, initialize_params_dnerf, load_dnerf_data
from external import (
    build_rotation,
    calc_psnr,
    calc_ssim,
    densify,
    update_params_and_optimizer,
)
from helpers import (
    l1_loss_v1,
    l1_loss_v2,
    o3d_knn,
    params2cpu,
    params2rendervar,
    quat_mult,
    save_params,
    setup_camera,
    weighted_l2_loss_v1,
    weighted_l2_loss_v2,
)
from hypernerf import (
    get_dataset_hypernerf,
    initialize_params_hypernerf,
    load_hypernerf_data,
)


def get_dataset(t, md, seq, data_dir):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(
            copy.deepcopy(
                Image.open(f"{data_dir}/{seq}/seg/{fn.replace('.jpg', '.png')}")
            )
        ).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, data_dir):
    init_pt_cld = np.load(f"{data_dir}/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {
        k: torch.nn.Parameter(
            torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
        )
        for k, v in params.items()
    }
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(
        np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1)
    )
    variables = {
        'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'scene_radius': scene_radius,
        'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0])
        .cuda()
        .float(),
        'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
    }
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    (
        im,
        radius,
        _,
    ) = Renderer(
        raster_settings=curr_data['cam']
    )(**rendervar)
    curr_id = curr_data['id']
    im = (
        torch.exp(params['cam_m'][curr_id])[:, None, None] * im
        + params['cam_c'][curr_id][:, None, None]
    )
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (
        1.0 - calc_ssim(im, curr_data['im'])
    )
    variables['means2D'] = rendervar[
        'means2D'
    ]  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    (
        seg,
        _,
        _,
    ) = Renderer(
        raster_settings=curr_data['cam']
    )(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (
        1.0 - calc_ssim(seg, curr_data['seg'])
    )

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (
            rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]
        ).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(
            curr_offset_in_prev_coord,
            variables["prev_offset"],
            variables["neighbor_weight"],
        )

        losses['rot'] = weighted_l2_loss_v2(
            rel_rot[variables["neighbor_indices"]],
            rel_rot[:, None],
            variables["neighbor_weight"],
        )

        curr_offset_mag = torch.sqrt((curr_offset**2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(
            curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"]
        )

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(
            bg_rot, variables["init_bg_rot"]
        )

        losses['soft_col_cons'] = l1_loss_v2(
            params['rgb_colors'], variables["prev_col"]
        )

    # Determine if we're using D-NeRF or HyperNeRF (indicated by random initialization)
    is_random_init = variables.get('is_random_init', False)

    if is_random_init:
        # Much lighter regularization for D-NeRF/HyperNeRF with random initialization
        loss_weights = {
            'im': 1.0,
            'seg': 3.0,
            'rigid': 0.1,  # Much lighter - allow points to move freely
            'rot': 0.1,  # Much lighter - allow rotations to change
            'iso': 0.05,  # Much lighter - allow distances to change
            'floor': 0.05,  # Much lighter - floor constraint barely enforced
            'bg': 0.5,  # Much lighter - background can move more
            'soft_col_cons': 0.01,
        }
    else:
        # Original weights for CMU dataset
        loss_weights = {
            'im': 1.0,
            'seg': 3.0,
            'rigid': 4.0,
            'rot': 4.0,
            'iso': 2.0,
            'floor': 2.0,
            'bg': 20.0,
            'soft_col_cons': 0.01,
        }
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(
        radius[seen], variables['max_2D_radius'][seen]
    )
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])

    # For random initialization (D-NeRF/HyperNeRF), use much gentler motion prediction
    is_random_init = variables.get('is_random_init', False)
    if is_random_init:
        # For the first few timesteps, don't predict motion at all - just keep current positions
        current_timestep = variables.get('current_timestep', 1)
        if current_timestep <= 3:  # First 3 timesteps: no motion prediction
            new_pts = pts.clone()
            new_rot = rot.clone()
            print(
                f"Timestep {current_timestep}: No motion prediction (stabilization phase)"
            )
        else:
            # After stabilization, use minimal motion
            motion_scale = 0.005 * variables['scene_radius']  # Even smaller motion
            random_motion = torch.randn_like(pts) * motion_scale
            new_pts = pts + random_motion

            # Minimal rotation change
            rot_noise = torch.randn_like(rot) * 0.005  # Smaller rotation noise
            new_rot = torch.nn.functional.normalize(rot + rot_noise)
    else:
        # Original linear extrapolation for CMU datasets
        new_pts = pts + (pts - variables["prev_pts"])
        new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(
        init_fg_pts.detach().cpu().numpy(), num_knn
    )
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = (
        torch.tensor(neighbor_indices).cuda().long().contiguous()
    )
    variables["neighbor_weight"] = (
        torch.tensor(neighbor_weight).cuda().float().contiguous()
    )
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(
        params['unnorm_rotations']
    ).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        (
            im,
            _,
            _,
        ) = Renderer(
            raster_settings=data['cam']
        )(**params2rendervar(params))
        curr_id = data['id']
        im = (
            torch.exp(params['cam_m'][curr_id])[:, None, None] * im
            + params['cam_c'][curr_id][:, None, None]
        )
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp, data_dir, output_dir, dataset_type="cmu"):
    if os.path.exists(f"{output_dir}/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    # Load data based on dataset type
    if dataset_type == "dnerf":
        md = load_dnerf_data(data_dir, seq)
        get_dataset_func = get_dataset_dnerf
        initialize_params_func = initialize_params_dnerf
        print(f"Loading D-NeRF dataset: {seq}")
    elif dataset_type == "hypernerf":
        md = load_hypernerf_data(data_dir, seq)
        get_dataset_func = get_dataset_hypernerf
        initialize_params_func = initialize_params_hypernerf
        print(f"Loading HyperNeRF dataset: {seq}")

        # Print HyperNeRF scene information
        scene_info = md.get('scene_info', {})
        if scene_info:
            print(f"HyperNeRF scene parameters:")
            print(f"  Scale: {scene_info.get('scale', 1.0)}")
            print(f"  Center: {scene_info.get('center', [0, 0, 0])}")
            print(
                f"  Near/Far: {scene_info.get('near', 1.0):.6f} / {scene_info.get('far', 100.0):.6f}"
            )
    else:
        md = json.load(open(f"{data_dir}/{seq}/train_meta.json", 'r'))  # metadata
        get_dataset_func = get_dataset
        initialize_params_func = initialize_params
        print(f"Loading CMU dataset: {seq}")

    num_timesteps = len(md['fn'])
    print(f"Training on {num_timesteps} timesteps")

    params, variables = initialize_params_func(seq, md, data_dir)
    optimizer = initialize_optimizer(params, variables)
    output_params = []

    for t in range(num_timesteps):
        dataset = get_dataset_func(t, md, seq, data_dir)
        todo_dataset = []
        is_initial_timestep = t == 0
        if not is_initial_timestep:
            # Track current timestep for motion prediction
            variables['current_timestep'] = t
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 3000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(
                params, curr_data, variables, is_initial_timestep
            )
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
    save_params(output_params, seq, exp, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Path to the data directory"
    )
    parser.add_argument("--exp-name", type=str, default="exp1", help="Experiment name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="basketball",
        help="Name of the dataset to use for training (e.g., basketball, boxes, etc.)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="cmu",
        choices=["cmu", "dnerf", "hypernerf"],
        help="Type of dataset format: 'cmu' for the current format, 'dnerf' for D-NeRF format, 'hypernerf' for HyperNeRF format",
    )
    args = parser.parse_args()
    train(
        args.dataset, args.exp_name, args.data_dir, args.output_dir, args.dataset_type
    )
