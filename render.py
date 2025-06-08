import json
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from PIL import Image

from helpers import setup_camera

# image size & camera clipping planes
w, h = 640, 360
near, far = 0.01, 100.0

# output method name
METHOD = "ours"


def load_scene_data(seq: str, exp: str, out_dir: Path) -> list[dict]:
    """
    Load per-timestep Gaussian params from params.npz,
    return a list of length T where each entry is the dict
    the renderer expects for that timestep.
    """
    npz_path = out_dir / exp / seq / "params.npz"
    raw = dict(np.load(npz_path))
    params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}

    T = params["means3D"].shape[0]
    scene = []
    for t in range(T):
        scene.append(
            {
                "means3D": params["means3D"][t],
                "colors_precomp": params["rgb_colors"][t],
                "rotations": torch.nn.functional.normalize(
                    params["unnorm_rotations"][t]
                ),
                "opacities": torch.sigmoid(params["logit_opacities"]),
                "scales": torch.exp(params["log_scales"]),
                "means2D": torch.zeros_like(params["means3D"][0], device="cuda"),
            }
        )
    return scene


def tensor_to_pil(im: torch.Tensor) -> Image.Image:
    """
    Convert a [C,H,W] float32 tensor with values in [0,1]
    into a PIL Image (uint8 RGB).
    """
    arr = (
        (torch.permute(im, (1, 2, 0)).cpu().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(arr)


def render_and_save(seq: str, exp: str, out_dir: Path, data_dir: Path):
    """
    For each (timestep, view) in train_meta.json:
      1. grab scene[t]
      2. render with that view's (k, w2c)
      3. save to .../test/METHOD/renders/<t>_<c>.png
      4. copy GT from data_dir/.../ims/<fn> → .../test/METHOD/gt/<t>_<c>.png
    """
    # 1) load scene data (length = # timesteps)
    scene = load_scene_data(seq, exp, out_dir)

    # 2) load metadata & build flat list of views
    meta_path = data_dir / seq / "train_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    views = []
    for t, (fns, ks, w2cs) in enumerate(zip(meta["fn"], meta["k"], meta["w2c"])):
        for c, fn in enumerate(fns):
            views.append(
                {
                    "t": t,
                    "c": c,
                    "fn": fn,
                    "k": np.array(ks[c]),
                    "w2c": np.array(w2cs[c]),
                }
            )

    # 3) prepare output folders
    base = out_dir / exp / seq / "test" / METHOD
    renders_dir = base / "renders"
    gt_dir = base / "gt"
    fps_path = base / "fps.txt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    timings = []
    # 4) render + copy GT for each view
    for view in views:
        ts = time.time()
        t, c, fn = view["t"], view["c"], view["fn"]
        data_vars = scene[t]  # pick the right timestep’s dict

        # build camera & render
        cam = setup_camera(w, h, view["k"], view["w2c"], near=near, far=far)
        with torch.no_grad():
            im, _, _ = Renderer(raster_settings=cam)(**data_vars)

        timings.append(time.time() - ts)
        # convert and save render
        img = tensor_to_pil(im)
        name = f"{t:04d}_{c:04d}.png"
        img.save(renders_dir / name)

        # copy the matching ground-truth image
        src = data_dir / seq / "ims" / fn
        dst = gt_dir / name
        shutil.copy(src, dst)

        print(f"Saved render → {renders_dir/name}    GT → {gt_dir/name}")

    # 5) save average FPS
    with open(fps_path, 'w') as f:
        total_time = sum(timings)
        f.write("0") if total_time < 1e-5 else f.write(str(len(views) / total_time))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render every (t, view) and pair with GT for evaluation"
    )
    parser.add_argument("--exp-name", type=str, default="pretrained")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="basketball",
        choices=["basketball", "boxes", "football", "juggle", "softball", "tennis"],
        help="Name of the dataset to use for training (e.g., basketball, boxes, etc.)",
    )
    args = parser.parse_args()

    print(f"\n=== Sequence: {args.dataset} ===")
    render_and_save(args.dataset, args.exp_name, args.output_dir, args.data_dir)
