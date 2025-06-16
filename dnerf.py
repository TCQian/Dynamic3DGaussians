import copy
import json
import os

import numpy as np
import torch
from PIL import Image

from helpers import o3d_knn, setup_camera


def load_dnerf_data(data_dir, seq):
    """Load D-NeRF dataset and convert to the expected format"""
    transforms_file = os.path.join(data_dir, seq, "transforms_train.json")
    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f"D-NeRF transforms file not found: {transforms_file}")

    print(f"Loading D-NeRF transforms from: {transforms_file}")

    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

    # Try to detect image dimensions from the first image
    first_frame = transforms['frames'][0] if transforms['frames'] else None
    if first_frame:
        file_path = first_frame['file_path'] + '.png'
        img_path = os.path.join(data_dir, seq, file_path)
        with Image.open(img_path) as img:
            w = img.width
            h = img.height
            print(f"Detected image dimensions from {img_path}: {w}x{h}")

    # Use defaults if image loading failed
    if 'w' not in locals() or 'h' not in locals():
        w = 800
        h = 600
        print(f"Using default dimensions: {w}x{h}")

    # Extract camera intrinsics (assuming all frames share the same camera)
    camera_angle_x = transforms.get('camera_angle_x', None)

    if camera_angle_x is not None:
        focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
        k = [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        print(
            f"Using camera_angle_x: {camera_angle_x:.4f} rad, focal length: {focal:.2f}"
        )
    else:
        # Try to get focal length directly
        fl_x = transforms.get('fl_x', None)
        fl_y = transforms.get('fl_y', None)
        cx = transforms.get('cx', w / 2)
        cy = transforms.get('cy', h / 2)

        if fl_x is not None and fl_y is not None:
            k = [[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]
            print(f"Using fl_x: {fl_x}, fl_y: {fl_y}")
        else:
            raise ValueError("Could not determine camera intrinsics from D-NeRF data")

    # Group frames by time
    frames_by_time = {}
    for frame in transforms['frames']:
        time = frame.get('time', 0.0)  # default to time 0 if not specified
        if time not in frames_by_time:
            frames_by_time[time] = []
        frames_by_time[time].append(frame)

    # Sort times
    sorted_times = sorted(frames_by_time.keys())
    print(
        f"Found {len(transforms['frames'])} frames across {len(sorted_times)} time steps"
    )
    print(f"Time range: {min(sorted_times):.3f} - {max(sorted_times):.3f}")

    # Convert to expected format
    fn = []
    w2c_list = []
    k_list = []

    for time in sorted_times:
        time_fn = []
        time_w2c = []
        time_k = []

        for frame in frames_by_time[time]:
            # Get filename with path
            file_path = frame['file_path'] + '.png'
            time_fn.append(file_path)

            # Convert transform matrix to w2c
            c2w = np.array(frame['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            time_w2c.append(w2c.tolist())

            # Camera intrinsics (same for all frames)
            time_k.append(k)

        fn.append(time_fn)
        w2c_list.append(time_w2c)
        k_list.append(time_k)

    md = {'fn': fn, 'w': w, 'h': h, 'k': k_list, 'w2c': w2c_list}

    return md


def get_dataset_dnerf(t, md, seq, data_dir):
    """Modified dataset loader for D-NeRF format"""
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        file_path = md['fn'][t][c]

        # Construct full image path
        img_path = os.path.join(data_dir, seq, file_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        im = np.array(copy.deepcopy(Image.open(img_path)))
        # Handle RGBA images
        if im.shape[-1] == 4:
            # Use alpha channel as mask and blend with white background
            alpha = im[:, :, 3:4] / 255.0
            im = im[:, :, :3] * alpha + 255 * (1 - alpha)
            im = im.astype(np.uint8)

        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        # For D-NeRF, we might not have segmentation masks
        # Create a dummy segmentation (all foreground)
        seg = torch.ones(h, w).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def initialize_params_dnerf(seq, md, data_dir):
    """Initialize parameters for D-NeRF - create point cloud from camera poses"""
    # Since D-NeRF might not have initial point cloud, create one from camera positions
    cam_centers = []
    for t in range(len(md['w2c'])):
        for c in range(len(md['w2c'][t])):
            w2c = np.array(md['w2c'][t][c])
            cam_center = np.linalg.inv(w2c)[:3, 3]
            cam_centers.append(cam_center)

    cam_centers = np.array(cam_centers)
    scene_center = np.mean(cam_centers, axis=0)
    scene_radius = np.max(np.linalg.norm(cam_centers - scene_center, axis=1))

    # Generate random points in a cube around the scene
    num_points = 2000
    print(f"Generating random point cloud ({num_points})...")

    # Create random points in a cube centered at scene_center
    points = np.random.random((num_points, 3)) * 2.6 - 1.3
    points = points * scene_radius * 0.8 + scene_center

    # Initialize with gray colors
    colors = np.ones((num_points, 3)) * 0.5

    init_pt_cld = np.column_stack(
        [
            points,
            colors,
            np.ones(num_points),  # segmentation (all foreground)
        ]
    )

    seg = init_pt_cld[:, 6]
    max_cams = max(50, len(cam_centers))
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
