import copy
import json
import os

import numpy as np
import torch
from PIL import Image

from helpers import o3d_knn, setup_camera


def load_hypernerf_data(data_dir, seq):
    """Load HyperNeRF dataset and convert to the expected format"""
    dataset_file = os.path.join(data_dir, seq, "dataset.json")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"HyperNeRF dataset.json not found: {dataset_file}")

    print(f"Loading HyperNeRF dataset from: {dataset_file}")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    # Load metadata for temporal information
    metadata_file = os.path.join(data_dir, seq, "metadata.json")
    metadata = {}
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"HyperNeRF metadata.json not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata with temporal information")

    # Load scene information
    scene_file = os.path.join(data_dir, seq, "scene.json")
    scene_info = {}
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"HyperNeRF scene.json not found: {scene_file}")

    with open(scene_file, 'r') as f:
        scene_info = json.load(f)
    print(f"Loaded scene.json with scale={scene_info.get('scale', 1.0)}")

    # Get frame IDs and train/test split
    all_ids = dataset.get('ids', [])
    train_ids = set(dataset.get('train_ids', all_ids))  # Use all if no train_ids
    val_ids = set(dataset.get('val_ids', []))

    print(f"Total frames: {len(all_ids)}")
    print(f"Training frames: {len(train_ids)}")
    print(f"Validation frames: {len(val_ids)}")

    # Look for camera folder with individual frame.json files
    camera_folder = os.path.join(data_dir, seq, "camera")
    if not os.path.exists(camera_folder):
        raise FileNotFoundError(f"HyperNeRF camera folder not found: {camera_folder}")

    # Try to detect image dimensions and find the best resolution
    first_train_id = next(iter(train_ids)) if train_ids else all_ids[0]

    # HyperNeRF stores images in different resolutions: 2x, 4x, 8x, 16x, using 4x here
    resolutions = ["4x"]  # ["2x", "4x", "8x", "16x"]
    img_path = None
    img_dir = None

    for res in resolutions:
        test_path = f"rgb/{res}/{first_train_id}.png"
        full_path = os.path.join(data_dir, seq, test_path)
        if os.path.exists(full_path):
            img_path = full_path
            img_dir = f"rgb/{res}"
            break

    if img_path is None:
        raise FileNotFoundError(
            f"Could not find image for frame {first_train_id} in any resolution in {data_dir}/{seq}"
        )

    # Get image dimensions
    with Image.open(img_path) as img:
        w, h = img.size
    print(f"Using resolution: {img_dir}")
    print(f"Detected image dimensions: {w}x{h} from {img_path}")

    # Load camera parameters from the first training frame
    camera_file = os.path.join(data_dir, seq, "camera", f"{first_train_id}.json")
    if os.path.exists(camera_file):
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)

        # Extract camera parameters from the individual camera file
        focal_length = camera_data.get('focal_length', w * 0.7)
        principal_point = camera_data.get('principal_point', [w / 2, h / 2])
        image_size = camera_data.get('image_size', [w, h])

        print(f"Original image size: {image_size}")
    else:
        raise FileNotFoundError(f"HyperNeRF camera file not found: {camera_file}")

    # Build intrinsic matrix
    k = [
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1],
    ]

    print(f"Camera parameters:")
    print(f"  Focal length: {focal_length}")
    print(f"  Principal point: {principal_point}")

    # Group frames by time_id from metadata for proper temporal ordering
    # This is crucial for HyperNeRF's temporal consistency
    frames_by_time = {}
    for frame_id in train_ids:
        if frame_id in metadata:
            time_id = metadata[frame_id]['time_id']
            if time_id not in frames_by_time:
                frames_by_time[time_id] = []
            frames_by_time[time_id].append(frame_id)
        else:
            print(f"Warning: No metadata for frame {frame_id}")

    # Sort by time_id for proper temporal ordering
    sorted_time_ids = sorted(frames_by_time.keys())
    print(f"Found {len(sorted_time_ids)} unique time steps")
    print(f"Time range: {min(sorted_time_ids)} - {max(sorted_time_ids)}")

    # Use temporal grouping
    frame_groups = [(time_id, frames_by_time[time_id]) for time_id in sorted_time_ids]

    # We treat each frame as a separate timestep
    fn = []
    w2c_list = []
    k_list = []

    # Calculate resolution scale for camera parameter adjustment
    resolution_scale = 1.0
    if "2x" in img_dir:
        resolution_scale = 0.5
    elif "4x" in img_dir:
        resolution_scale = 0.25
    elif "8x" in img_dir:
        resolution_scale = 0.125
    elif "16x" in img_dir:
        resolution_scale = 0.0625

    for time_id, frame_ids in frame_groups:
        # For now, we'll still treat each frame as its own timestep
        # But we could group multiple frames per timestep if needed
        for frame_id in frame_ids:
            time_fn = []
            time_w2c = []
            time_k = []

            # Determine image path using the detected resolution
            file_path = f"{img_dir}/{frame_id}.png"
            full_path = os.path.join(data_dir, seq, file_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Image file not found: {full_path}")

            time_fn.append(file_path)

            # Load camera pose for this frame from individual camera file
            camera_file = os.path.join(data_dir, seq, "camera", f"{frame_id}.json")
            with open(camera_file, 'r') as f:
                frame_camera = json.load(f)

            # Extract pose from HyperNeRF camera format
            orientation = np.array(frame_camera['orientation'])  # 3x3 rotation matrix
            position = np.array(frame_camera['position'])  # 3D position

            # Apply scene scaling if available
            if 'scale' in scene_info:
                position = position * scene_info['scale']

            # Build camera-to-world matrix
            c2w = np.eye(4)
            c2w[:3, :3] = orientation
            c2w[:3, 3] = position

            # Convert to world-to-camera
            w2c = np.linalg.inv(c2w)

            # Use frame-specific camera parameters
            frame_focal = frame_camera.get('focal_length', focal_length)
            frame_pp = frame_camera.get('principal_point', principal_point)

            # Apply resolution scaling
            frame_focal *= resolution_scale
            frame_pp = [p * resolution_scale for p in frame_pp]

            frame_k = [
                [frame_focal, 0, frame_pp[0]],
                [0, frame_focal, frame_pp[1]],
                [0, 0, 1],
            ]

            time_k.append(frame_k)
            time_w2c.append(w2c.tolist())
            fn.append(time_fn)
            w2c_list.append(time_w2c)
            k_list.append(time_k)

    print(f"Loaded {len(fn)} timesteps for training")

    # Store scene information in metadata for use in camera setup
    md = {
        'fn': fn,
        'w': w,
        'h': h,
        'k': k_list,
        'w2c': w2c_list,
        'scene_info': scene_info,  # Add scene info for near/far planes
    }
    return md


def get_dataset_hypernerf(t, md, seq, data_dir):
    """Modified dataset loader for HyperNeRF format"""
    dataset = []

    # Get scene information for camera setup
    scene_info = md.get('scene_info', {})
    near = scene_info.get('near', 1.0)
    far = scene_info.get('far', 100.0)

    print(f"Using camera planes: near={near:.6f}, far={far:.6f}")

    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=near, far=far)
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

        # For HyperNeRF, we might not have segmentation masks
        # Create a dummy segmentation (all foreground)
        seg = torch.ones(h, w).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def initialize_params_hypernerf(seq, md, data_dir):
    """Initialize parameters for HyperNeRF - load point cloud from points.npy"""
    # Get scene information
    scene_info = md.get('scene_info', {})
    scene_scale = scene_info.get('scale', 1.0)
    scene_center = np.array(scene_info.get('center', [0, 0, 0]))

    print(f"Scene info: scale={scene_scale}, center={scene_center}")

    # Load point cloud from points.npy
    points_file = os.path.join(data_dir, seq, "points.npy")
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"HyperNeRF points.npy not found: {points_file}")

    print(f"Loading point cloud from: {points_file}")
    points = np.load(points_file)

    # HyperNeRF points.npy typically contains just XYZ coordinates
    if points.shape[1] == 3:
        # Initialize with gray colors and create foreground/background split
        colors = np.ones((points.shape[0], 3)) * 0.5

        # Create segmentation based on point distribution
        # Points closer to scene center are more likely to be foreground
        if np.any(scene_center != 0):
            computed_center = scene_center
        else:
            computed_center = np.mean(points, axis=0)

        distances_to_center = np.linalg.norm(points - computed_center, axis=1)

        # Use a probability based on distance - closer points more likely to be foreground
        scene_radius = np.max(distances_to_center)
        fg_prob = np.exp(-distances_to_center / (scene_radius * 0.5))
        seg = (np.random.random(points.shape[0]) < fg_prob).astype(float)

        # Ensure at least 60% are foreground for better training
        if seg.mean() < 0.6:
            n_fg_needed = int(0.6 * points.shape[0]) - int(seg.sum())
            bg_indices = np.where(seg == 0)[0]
            if len(bg_indices) >= n_fg_needed:
                seg[bg_indices[:n_fg_needed]] = 1.0

        init_pt_cld = np.column_stack([points, colors, seg])
        print(f"Loaded {points.shape[0]} points from file")
    else:
        # If points file has more columns, assume it's already formatted
        init_pt_cld = points
        seg = (
            init_pt_cld[:, 6]
            if init_pt_cld.shape[1] > 6
            else np.ones(init_pt_cld.shape[0]) * 0.8
        )
        print(
            f"Loaded {points.shape[0]} points with {points.shape[1]} columns from file"
        )

    # Get camera centers for scene radius calculation
    cam_centers = []
    for t in range(len(md['w2c'])):
        for c in range(len(md['w2c'][t])):
            w2c = np.array(md['w2c'][t][c])
            cam_center = np.linalg.inv(w2c)[:3, 3]
            cam_centers.append(cam_center)
    cam_centers = np.array(cam_centers)

    # Calculate max cameras per timestep (not total across all timesteps)
    max_cams = max(len(md['fn'][t]) for t in range(len(md['fn'])))
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
        'is_hypernerf': True,
    }
    return params, variables
