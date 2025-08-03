#!/usr/bin/env python3
"""
Preprocessing script to convert DyNeRF format to Dynamic3DGaussians CMU format

This script converts the original DyNeRF dataset format (poses_bounds.npy + videos)
to the CMU panoptic format expected by Dynamic3DGaussians.
"""

import numpy as np
import cv2
import json
import os
import shutil
from PIL import Image
from tqdm import tqdm
import argparse
from scipy.spatial import cKDTree
import open3d as o3d


def load_poses_bounds(poses_bounds_path):
    """Load and parse poses_bounds.npy file"""
    poses_bounds = np.load(poses_bounds_path)
    print(f"Loaded poses_bounds.npy with shape: {poses_bounds.shape}")
    
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
    bounds = poses_bounds[:, 15:17]  # (N, 2)
    
    # Extract camera parameters from poses
    # poses[:, :, :3] = rotation matrix
    # poses[:, :, 3] = translation
    # poses[:, :, 4] = [height, width, focal_length]
    
    hwf = poses[0, :, 4]  # height, width, focal_length
    h, w, f = int(hwf[0]), int(hwf[1]), hwf[2]
    
    print(f"Camera parameters: H={h}, W={w}, f={f}")
    
    return poses, bounds, h, w, f


def poses_to_matrices(poses, h, w, f):
    """Convert LLFF poses to camera matrices"""
    n_cams = poses.shape[0]
    
    # Camera intrinsics (assuming principal point at center)
    K = np.array([
        [f, 0, w/2],
        [0, f, h/2], 
        [0, 0, 1]
    ])
    
    cameras = []
    for i in range(n_cams):
        # Extract rotation and translation
        R = poses[i, :3, :3]  # 3x3 rotation
        t = poses[i, :3, 3]   # 3x1 translation
        
        # Convert from camera-to-world to world-to-camera
        # c2w -> w2c: R_w2c = R_c2w.T, t_w2c = -R_c2w.T @ t_c2w
        R_w2c = R.T
        t_w2c = -R_w2c @ t
        
        # Create 4x4 transformation matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        
        cameras.append({
            'K': K.copy(),
            'w2c': w2c,
            'c2w': np.linalg.inv(w2c)
        })
    
    return cameras


def extract_frames_from_videos(data_dir, seq, output_dir, target_size=(640, 360), max_frames=150):
    """Extract frames from video files"""
    seq_path = os.path.join(data_dir, seq)
    output_ims_dir = os.path.join(output_dir, seq, "ims")
    os.makedirs(output_ims_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for f in os.listdir(seq_path):
        if f.endswith('.mp4') and f.startswith('cam'):
            cam_id = int(f.replace('cam', '').replace('.mp4', ''))
            video_files.append((cam_id, f))
    
    video_files.sort()  # Sort by camera ID
    print(f"Found {len(video_files)} video files")
    
    all_frames = {}
    fps_info = {}
    
    for cam_id, video_file in tqdm(video_files, desc="Extracting frames"):
        video_path = os.path.join(seq_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_info[cam_id] = fps
        
        frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_filename = f"{frame_count:06d}_{cam_id:02d}.jpg"
            frame_path = os.path.join(output_ims_dir, frame_filename)
            Image.fromarray(frame_rgb).save(frame_path)
            
            frames.append(frame_filename)
            frame_count += 1
        
        cap.release()
        all_frames[cam_id] = frames
        print(f"  Camera {cam_id:02d}: extracted {len(frames)} frames (fps: {fps:.2f})")
    
    return all_frames, fps_info


def create_segmentation_masks(data_dir, seq, output_dir, all_frames, background_frame_idx=0):
    """Create foreground-background segmentation masks using frame differencing"""
    seq_path = os.path.join(data_dir, seq)
    output_seg_dir = os.path.join(output_dir, seq, "seg")
    os.makedirs(output_seg_dir, exist_ok=True)
    
    ims_dir = os.path.join(output_dir, seq, "ims")
    
    print("Creating segmentation masks using frame differencing...")
    
    # Load background reference frames (first frame from each camera)
    background_frames = {}
    for cam_id, frames in all_frames.items():
        if len(frames) > background_frame_idx:
            bg_frame_path = os.path.join(ims_dir, frames[background_frame_idx])
            bg_frame = np.array(Image.open(bg_frame_path)).astype(np.float32)
            background_frames[cam_id] = bg_frame
    
    # Generate masks for all frames
    for cam_id, frames in tqdm(all_frames.items(), desc="Creating masks"):
        if cam_id not in background_frames:
            continue
            
        bg_frame = background_frames[cam_id]
        
        for frame_filename in frames:
            frame_path = os.path.join(ims_dir, frame_filename)
            current_frame = np.array(Image.open(frame_path)).astype(np.float32)
            
            # Frame differencing
            diff = np.abs(current_frame - bg_frame).mean(axis=2)
            
            # Thresholding to create binary mask
            threshold = 30.0  # Adjust based on your data
            mask = (diff > threshold).astype(np.float32)
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Save mask
            mask_filename = frame_filename.replace('.jpg', '.png')
            mask_path = os.path.join(output_seg_dir, mask_filename)
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)


def create_initial_point_cloud(output_dir, seq, cameras, all_frames, subsample_factor=2):
    """Create initial point cloud using depth estimation or random sampling"""
    print("Creating initial point cloud...")
    
    # Since we don't have depth cameras, we'll create a simple point cloud
    # by sampling points in the scene volume based on camera positions
    
    # Get camera centers
    cam_centers = []
    for cam_info in cameras:
        center = cam_info['c2w'][:3, 3]
        cam_centers.append(center)
    cam_centers = np.array(cam_centers)
    
    # Define scene bounds based on camera positions
    scene_center = np.mean(cam_centers, axis=0)
    scene_radius = np.max(np.linalg.norm(cam_centers - scene_center, axis=1)) * 0.5
    
    # Create a grid of points in the scene
    n_points_per_dim = 32
    x_range = np.linspace(scene_center[0] - scene_radius, scene_center[0] + scene_radius, n_points_per_dim)
    y_range = np.linspace(scene_center[1] - scene_radius, scene_center[1] + scene_radius, n_points_per_dim)
    z_range = np.linspace(scene_center[2] - scene_radius, scene_center[2] + scene_radius, n_points_per_dim)
    
    points = []
    colors = []
    seg_labels = []
    
    ims_dir = os.path.join(output_dir, seq, "ims")
    seg_dir = os.path.join(output_dir, seq, "seg")
    
    # Sample points and get colors from camera projections
    for x in tqdm(x_range[::subsample_factor], desc="Sampling points"):
        for y in y_range[::subsample_factor]:
            for z in z_range[::subsample_factor]:
                point_3d = np.array([x, y, z])
                
                # Project to cameras and get color
                best_color = None
                best_seg = 0
                min_distance = float('inf')
                
                for i, cam_info in enumerate(cameras):
                    if i not in all_frames or len(all_frames[i]) == 0:
                        continue
                        
                    # Project 3D point to camera
                    point_3d_homo = np.append(point_3d, 1)
                    point_2d_homo = cam_info['K'] @ (cam_info['w2c'][:3] @ point_3d_homo)
                    
                    if point_2d_homo[2] <= 0:  # Behind camera
                        continue
                        
                    point_2d = point_2d_homo[:2] / point_2d_homo[2]
                    u, v = int(point_2d[0]), int(point_2d[1])
                    
                    # Check if projection is within image bounds
                    if 0 <= u < 640 and 0 <= v < 360:
                        distance = np.linalg.norm(cam_info['c2w'][:3, 3] - point_3d)
                        
                        if distance < min_distance:
                            # Load first frame to get color
                            frame_path = os.path.join(ims_dir, all_frames[i][0])
                            seg_path = os.path.join(seg_dir, all_frames[i][0].replace('.jpg', '.png'))
                            
                            if os.path.exists(frame_path):
                                img = np.array(Image.open(frame_path))
                                color = img[v, u] / 255.0
                                
                                seg = 0
                                if os.path.exists(seg_path):
                                    seg_img = np.array(Image.open(seg_path))
                                    seg = 1 if seg_img[v, u] > 128 else 0
                                
                                best_color = color
                                best_seg = seg
                                min_distance = distance
                
                if best_color is not None:
                    points.append(point_3d)
                    colors.append(best_color)
                    seg_labels.append(best_seg)
    
    if len(points) == 0:
        print("Warning: No valid points found. Creating a simple point cloud.")
        # Create a simple point cloud around scene center
        n_points = 1000
        points = np.random.normal(scene_center, scene_radius/3, (n_points, 3))
        colors = np.random.rand(n_points, 3)
        seg_labels = np.zeros(n_points)
    
    points = np.array(points)
    colors = np.array(colors)
    seg_labels = np.array(seg_labels)
    
    # Combine points, colors, and segmentation
    init_pt_cld = np.column_stack([points, colors, seg_labels])
    
    # Save point cloud
    output_path = os.path.join(output_dir, seq, "init_pt_cld.npz")
    np.savez(output_path, data=init_pt_cld)
    
    print(f"Created initial point cloud with {len(points)} points")
    print(f"Foreground points: {np.sum(seg_labels)}, Background points: {np.sum(1-seg_labels)}")


def create_metadata_json(output_dir, seq, cameras, all_frames, fps_info, target_size=(640, 360)):
    """Create train_meta.json in CMU format"""
    print("Creating metadata JSON...")
    
    metadata = {
        'w': target_size[0],  # width
        'h': target_size[1],  # height
        'fn': [],  # filenames per timestep
        'k': [],   # intrinsics per timestep
        'w2c': []  # world-to-camera matrices per timestep
    }
    
    # Determine number of timesteps (frames)
    max_frames = max(len(frames) for frames in all_frames.values())
    
    # Split cameras into train and test (following PanopticSports split)
    all_cam_ids = sorted(all_frames.keys())
    test_cam_ids = set([0, 10, 15, 19])
    assert test_cam_ids in all_cam_ids, f"Camera ids {test_cam_ids} are not found"
    train_cam_ids = [cam_id for cam_id in all_cam_ids if cam_id not in test_cam_ids]
    
    print(f"Train cameras: {train_cam_ids}")
    print(f"Test cameras: {list(test_cam_ids)}")
    
    for t in range(max_frames):
        frame_filenames = []
        frame_intrinsics = []
        frame_w2c = []
        
        for cam_id in train_cam_ids:
            if cam_id in all_frames and t < len(all_frames[cam_id]):
                frame_filenames.append(all_frames[cam_id][t])
                frame_intrinsics.append(cameras[cam_id]['K'].tolist())
                frame_w2c.append(cameras[cam_id]['w2c'].tolist())
        
        if frame_filenames:  # Only add timestep if we have at least one frame
            metadata['fn'].append(frame_filenames)
            metadata['k'].append(frame_intrinsics)
            metadata['w2c'].append(frame_w2c)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, seq, "train_meta.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata for {len(metadata['fn'])} timesteps with {len(train_cam_ids)} training cameras")


def preprocess_dynerf_sequence(data_dir, seq, output_dir, target_size=(640, 360), max_frames=150):
    """
    Main preprocessing function to convert DyNeRF format to CMU format
    
    Args:
        data_dir: Directory containing the original DyNeRF data
        seq: Sequence name (e.g., 'cut_roasted_beef')
        output_dir: Output directory for processed data
        target_size: Target image size (width, height)
        max_frames: Maximum number of frames to extract
    """
    print(f"Preprocessing DyNeRF sequence: {seq}")
    print(f"Input directory: {data_dir}/{seq}")
    print(f"Output directory: {output_dir}/{seq}")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, seq), exist_ok=True)
    
    # Step 1: Load poses and bounds
    poses_bounds_path = os.path.join(data_dir, seq, "poses_bounds.npy")
    poses, bounds, h, w, f = load_poses_bounds(poses_bounds_path)
    
    # Step 2: Convert poses to camera matrices
    cameras = poses_to_matrices(poses, h, w, f)
    
    # Step 3: Extract frames from videos
    all_frames, fps_info = extract_frames_from_videos(
        data_dir, seq, output_dir, target_size, max_frames
    )
    
    # Step 4: Create segmentation masks
    create_segmentation_masks(data_dir, seq, output_dir, all_frames)
    
    # Step 5: Create initial point cloud
    create_initial_point_cloud(output_dir, seq, cameras, all_frames)
    
    # Step 6: Create metadata JSON
    create_metadata_json(output_dir, seq, cameras, all_frames, fps_info, target_size)
    
    print(f"✓ Preprocessing complete! Processed data saved to: {output_dir}/{seq}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DyNeRF data for Dynamic3DGaussians")
    parser.add_argument("--data-dir", type=str, default=".", 
                       help="Directory containing the original DyNeRF data")
    parser.add_argument("--seq", type=str, default="cut_roasted_beef",
                       help="Sequence name to preprocess")
    parser.add_argument("--output-dir", type=str, default="./processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--width", type=int, default=640,
                       help="Target image width")
    parser.add_argument("--height", type=int, default=360,
                       help="Target image height")
    parser.add_argument("--max-frames", type=int, default=150,
                       help="Maximum number of frames to extract")
    
    args = parser.parse_args()
    
    preprocess_dynerf_sequence(
        args.data_dir, 
        args.seq, 
        args.output_dir,
        target_size=(args.width, args.height),
        max_frames=args.max_frames
    ) 