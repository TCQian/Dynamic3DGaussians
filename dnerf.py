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

    else:
        raise ValueError("Could not determine image dimensions from D-NeRF data")

    # Extract camera intrinsics (assuming all frames share the same camera)
    camera_angle_x = transforms.get('camera_angle_x', None)

    if camera_angle_x:
        focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
        k = [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        print(
            f"Using camera_angle_x: {camera_angle_x:.4f} rad, focal length: {focal:.2f}"
        )
    else:
        raise ValueError("Could not determine camera intrinsics from D-NeRF data")

    # Get all frames and create proper time mapping like the reference
    all_frames = transforms['frames']

    # Create time mapper - map unique time values to sequential indices
    unique_times = sorted(list(set(frame.get('time', 0.0) for frame in all_frames)))
    time_mapper = {time_val: idx for idx, time_val in enumerate(unique_times)}

    print(
        f"Found {len(all_frames)} frames across {len(unique_times)} unique time values"
    )
    print(f"Time range: {min(unique_times):.3f} - {max(unique_times):.3f}")

    # Group frames by mapped time indices
    frames_by_timestep = {}
    for frame in all_frames:
        frame_time = frame.get('time', 0.0)
        timestep = time_mapper[frame_time]
        if timestep not in frames_by_timestep:
            frames_by_timestep[timestep] = []
        frames_by_timestep[timestep].append(frame)

    # Convert to expected format
    fn = []
    w2c_list = []
    k_list = []

    for timestep in sorted(frames_by_timestep.keys()):
        time_fn = []
        time_w2c = []
        time_k = []

        for frame in frames_by_timestep[timestep]:
            # Get filename with path
            file_path = frame['file_path'] + '.png'
            time_fn.append(file_path)

            # Convert transform matrix to w2c following reference implementation
            c2w = np.array(frame['transform_matrix'])
            # Apply the same transformations as reference
            matrix = np.linalg.inv(c2w)
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            # Reconstruct w2c matrix
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            time_w2c.append(w2c.tolist())

            # Camera intrinsics (same for all frames)
            time_k.append(k)

        fn.append(time_fn)
        w2c_list.append(time_w2c)
        k_list.append(time_k)

    md = {'fn': fn, 'w': w, 'h': h, 'k': k_list, 'w2c': w2c_list}
    print(f"Loaded {len(fn)} timesteps for D-NeRF training")

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

        # Load and process image following reference implementation
        image = Image.open(img_path)
        im_data = np.array(image.convert("RGBA"))

        # Handle alpha blending like reference (white background for D-NeRF)
        bg = np.array([1, 1, 1])  # white background
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        im = np.array(arr * 255.0, dtype=np.uint8)

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

    # Use consistent scene radius calculation throughout
    scene_radius = np.max(np.linalg.norm(cam_centers - scene_center, axis=1))

    print(
        f"Camera positions range: {cam_centers.min(axis=0)} to {cam_centers.max(axis=0)}"
    )
    print(f"Scene center: {scene_center}")
    print(f"Scene radius: {scene_radius}")

    # Generate more points for better coverage of dynamic scenes
    num_points = 5000  # Increased from 3000
    print(f"Generating random point cloud ({num_points})...")

    # Create better distributed points using multiple strategies
    # 1) Points distributed in a sphere around scene center (better than cube)
    points_sphere = np.random.randn(num_points // 2, 3)
    points_sphere = points_sphere / np.linalg.norm(points_sphere, axis=1, keepdims=True)
    # Distribute points within scene radius with more density near center
    radii = (
        np.random.beta(2, 5, num_points // 2) * scene_radius * 1.2
    )  # Beta distribution favors smaller radii
    points_sphere = points_sphere * radii.reshape(-1, 1) + scene_center

    # 2) Points distributed along camera ray directions for better scene coverage
    ray_points = []
    n_ray_points = num_points // 2
    for i in range(min(len(cam_centers), n_ray_points)):
        cam_pos = cam_centers[i % len(cam_centers)]
        # Create points along the ray from camera to scene center
        direction = scene_center - cam_pos
        direction = direction / np.linalg.norm(direction)
        # Distribute points at various distances along the ray
        distances = (
            np.random.uniform(0.1, 2.0, n_ray_points // len(cam_centers) + 1)
            * scene_radius
        )
        for dist in distances[: n_ray_points // len(cam_centers) + 1]:
            if len(ray_points) < n_ray_points:
                point = cam_pos + direction * dist
                ray_points.append(point)

    ray_points = np.array(ray_points[:n_ray_points])

    # Combine both point distributions
    if len(ray_points) > 0:
        points = np.vstack([points_sphere, ray_points])
    else:
        points = points_sphere

    # Trim to exact number needed
    points = points[:num_points]

    # Initialize with more varied colors (not just gray)
    colors = np.random.uniform(0.3, 0.7, (num_points, 3))  # Varied colors

    # Create a more sophisticated foreground/background split
    distances_to_center = np.linalg.norm(points - scene_center, axis=1)

    # Use sigmoid-based probability for smoother transition
    # Points closer to cameras and scene center are more likely to be foreground
    min_cam_dist = np.min(
        [np.linalg.norm(points - cam_pos, axis=1) for cam_pos in cam_centers], axis=0
    )

    # Combine center distance and camera distance for better fg/bg classification
    center_prob = 1 / (
        1 + np.exp((distances_to_center - scene_radius * 0.6) / (scene_radius * 0.1))
    )
    camera_prob = 1 / (
        1 + np.exp((min_cam_dist - scene_radius * 0.8) / (scene_radius * 0.1))
    )
    fg_prob = 0.7 * center_prob + 0.3 * camera_prob

    seg = (np.random.random(num_points) < fg_prob).astype(float)

    # Ensure at least 70% are foreground for better dynamic scene representation
    if seg.mean() < 0.7:
        n_fg_needed = int(0.7 * num_points) - int(seg.sum())
        bg_indices = np.where(seg == 0)[0]
        if len(bg_indices) >= n_fg_needed:
            # Prioritize points closer to scene center for foreground
            bg_distances = distances_to_center[bg_indices]
            closest_bg_indices = bg_indices[np.argsort(bg_distances)[:n_fg_needed]]
            seg[closest_bg_indices] = 1.0

    print(f"Foreground ratio: {seg.mean():.3f}")

    init_pt_cld = np.column_stack([points, colors, seg])

    # Calculate max cameras per timestep based on actual data
    max_cams_per_timestep = max(len(md['fn'][t]) for t in range(len(md['fn'])))
    max_cams = max(10, max_cams_per_timestep)  # At least 10 cameras
    print(f"Max cameras per timestep: {max_cams}")

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

    # Use consistent scene radius calculation (same as above)
    scene_radius_final = 1.1 * np.max(
        np.linalg.norm(cam_centers - scene_center, axis=1)
    )
    variables = {
        'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'scene_radius': scene_radius_final,
        'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0])
        .cuda()
        .float(),
        'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
        'is_random_init': True,  # Mark as using random initialization
    }
    return params, variables
