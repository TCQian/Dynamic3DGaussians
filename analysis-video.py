import argparse
import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np


def natural_sort_key(text):
    """
    Sort function that handles numeric sequences in filenames naturally.
    For example: img1.png, img2.png, img10.png instead of img1.png, img10.png, img2.png
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]


def get_image_files(folder_path, extensions=None):
    """
    Get all image files from the specified folder.

    Args:
        folder_path (str): Path to the folder containing images
        extensions (list): List of image extensions to include (default: common formats)

    Returns:
        list: Sorted list of image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern_upper = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper))

    # Remove duplicates and sort naturally
    image_files = list(set(image_files))
    image_files.sort(key=natural_sort_key)

    return image_files


def get_video_dimensions(image_files, target_size=None):
    """
    Determine the video dimensions based on the first image or target size.

    Args:
        image_files (list): List of image file paths
        target_size (tuple): Optional target size (width, height)

    Returns:
        tuple: (width, height) for the video
    """
    if target_size:
        return target_size

    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        raise ValueError(f"Could not read the first image: {image_files[0]}")

    height, width = first_img.shape[:2]
    return width, height


def create_video_from_images(
    folder_path,
    output_path,
    fps=30,
    target_size=None,
    codec='mp4v',
    extensions=None,
    verbose=True,
):
    """
    Create a video from images in a folder.

    Args:
        folder_path (str): Path to folder containing images
        output_path (str): Path for output video file
        fps (int): Frames per second for the output video
        target_size (tuple): Optional target size (width, height) to resize images
        codec (str): Video codec to use ('mp4v', 'XVID', etc.)
        extensions (list): List of image extensions to include
        verbose (bool): Whether to print progress information

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get all image files
        image_files = get_image_files(folder_path, extensions)

        if not image_files:
            print(f"No image files found in {folder_path}")
            return False

        if verbose:
            print(f"Found {len(image_files)} images")
            print(f"First image: {os.path.basename(image_files[0])}")
            print(f"Last image: {os.path.basename(image_files[-1])}")

        # Get video dimensions
        width, height = get_video_dimensions(image_files, target_size)

        if verbose:
            print(f"Video dimensions: {width}x{height}")
            print(f"Frame rate: {fps} FPS")
            print(f"Codec: {codec}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return False

        # Process each image
        for i, img_path in enumerate(image_files):
            if verbose and (i + 1) % 10 == 0:
                print(
                    f"Processing image {i + 1}/{len(image_files)}: {os.path.basename(img_path)}"
                )

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping...")
                continue

            # Resize image if necessary
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))

            # Write frame to video
            video_writer.write(img)

        # Release video writer
        video_writer.release()

        if verbose:
            print(f"Video created successfully: {output_path}")
            print(f"Total frames: {len(image_files)}")
            print(f"Duration: {len(image_files) / fps:.2f} seconds")

        return True

    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Create a video from images in a folder'
    )
    parser.add_argument('input_folder', help='Path to folder containing images')
    parser.add_argument(
        '-o',
        '--output',
        default='video.mp4',
        help='Output video file path (default: output_video.mp4)',
    )
    parser.add_argument(
        '--fps', type=int, default=30, help='Frames per second (default: 30)'
    )
    parser.add_argument('--width', type=int, help='Target width (optional)')
    parser.add_argument('--height', type=int, help='Target height (optional)')
    parser.add_argument('--codec', default='mp4v', help='Video codec (default: mp4v)')
    parser.add_argument(
        '--extensions',
        nargs='+',
        help='Image extensions to include (e.g., --extensions *.jpg *.png)',
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return

    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a directory")
        return

    # Set target size if both width and height are provided
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)
    elif args.width or args.height:
        print(
            "Warning: Both width and height must be specified together. Ignoring size parameters."
        )

    # Create video
    success = create_video_from_images(
        folder_path=args.input_folder,
        output_path=args.output,
        fps=args.fps,
        target_size=target_size,
        codec=args.codec,
        extensions=args.extensions,
        verbose=not args.quiet,
    )

    if success:
        print(f"\n✅ Video creation completed successfully!")
        print(f"Output: {args.output}")
    else:
        print(f"\n❌ Video creation failed!")


if __name__ == "__main__":
    main()
