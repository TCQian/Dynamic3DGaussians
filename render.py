from argparse import ArgumentParser
import torch
import numpy as np
import open3d as o3d
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera
import os

w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def load_scene_data(seq, exp, output_dir):
    params = dict(np.load(f"{output_dir}/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    rendervar = {
        'means3D': params['means3D'][0],
        'colors_precomp': params['rgb_colors'][0],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][0]),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
    }
    return rendervar

def render_image(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, _ = Renderer(raster_settings=cam)(**timestep_data)
        return im

def tensor_to_image(im):
    im_np = torch.permute(im, (1, 2, 0)).detach().cpu().numpy()
    im_np = np.clip(im_np * 255, 0, 255).astype(np.uint8)
    im_np = np.ascontiguousarray(im_np)
    return o3d.geometry.Image(im_np)

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="pretrained", help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="./output", help="Path to the output directory")
    args = parser.parse_args()
    from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord


    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    for seq in sequences:
        print(f"Rendering {seq}...")
        scene_data = load_scene_data(seq, args.exp_name, args.output_dir)
        w2c, k = init_camera()
        im = render_image(w2c, k, scene_data)

        # Convert tensor image to Open3D image
        img = tensor_to_image(im)

        # Save using OffscreenRenderer (just to initialize context safely)
        renderer = OffscreenRenderer(int(w * view_scale), int(h * view_scale))
        img_path = f"{args.output_dir}/{args.exp_name}/{seq}_frame0.png"
        o3d.io.write_image(img_path, img)
        del renderer
        print(f"Saved to {img_path}")

if __name__ == "__main__":
    main()
