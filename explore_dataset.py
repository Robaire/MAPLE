#!/usr/bin/env python3
"""
Flexible Dataset Explorer for Multiple LAC (.lac) files
This script processes multiple datasets and organizes them in a clean folder structure.
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import glob

# Import the lac-data classes
from lac_data import FrameDataReader, CameraDataReader


class DatasetExporter:
    """Handles exporting LAC datasets to organized folder structure."""

    def __init__(self, base_output_dir="datasets"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)

    def export_dataset(self, lac_file_path, max_images_per_camera=10):
        """
        Export a single LAC dataset to organized folder structure.

        Args:
            lac_file_path (str): Path to the .lac file

        Returns:
            dict: Paths to exported files
        """

        # Get dataset name from file path
        dataset_name = os.path.splitext(os.path.basename(lac_file_path))[0]

        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 60)

        # Create dataset directory
        dataset_dir = os.path.join(self.base_output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        try:
            # Initialize readers
            frame_reader = FrameDataReader(lac_file_path)
            camera_reader = CameraDataReader(lac_file_path)

            # Get basic information
            total_frames = len(frame_reader.frames)
            cameras = camera_reader.get_cameras()
            file_size = os.path.getsize(lac_file_path)

            print(f"Total frames: {total_frames}")
            print(f"Available cameras: {cameras}")
            print(f"File size: {file_size / (1024*1024):.2f} MB")

            # 1. Export frame data to CSV
            frame_data_path = os.path.join(dataset_dir, "frame_data.csv")
            frame_reader.frames.to_csv(frame_data_path, index=False)
            print(f"Frame data exported: {frame_data_path}")

            # 2. Export frame data to NumPy array
            np_path = os.path.join(dataset_dir, "frame_data.npy")
            frame_array = frame_reader.frames.values
            np.save(np_path, frame_array)
            print(f"NumPy array exported: {np_path}")

            # 3. Export column names for reference
            columns_path = os.path.join(dataset_dir, "columns.txt")
            with open(columns_path, "w") as f:
                for i, col in enumerate(frame_reader.frames.columns):
                    f.write(f"{i}: {col}\n")
            print(f"Column names exported: {columns_path}")

            # 4. Create metadata
            metadata = {
                "dataset_name": dataset_name,
                "source_file": lac_file_path,
                "export_date": datetime.now().isoformat(),
                "total_frames": total_frames,
                "file_size_mb": file_size / (1024 * 1024),
                "shape": frame_array.shape,
                "columns": list(frame_reader.frames.columns),
                "dtypes": {
                    col: str(dtype) for col, dtype in frame_reader.frames.dtypes.items()
                },
                "cameras": cameras,
                "initial_data": {},
            }

            # Add initial data if available
            if hasattr(frame_reader, "initial") and frame_reader.initial is not None:
                for key, value in frame_reader.initial.items():
                    if isinstance(value, (int, float, str, list)):
                        metadata["initial_data"][key] = value
                    elif isinstance(value, np.ndarray):
                        metadata["initial_data"][key] = value.tolist()

            # Save metadata
            metadata_path = os.path.join(dataset_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata exported: {metadata_path}")

            # 5. Export sample images
            sample_images_dir = os.path.join(dataset_dir, "sample_images")
            os.makedirs(sample_images_dir, exist_ok=True)

            images_exported = self.export_sample_images(
                camera_reader,
                cameras,
                sample_images_dir,
                total_frames,
                max_images_per_camera,
            )

            print(
                f"Sample images exported: {images_exported} images to {sample_images_dir}"
            )

            # 6. Create summary file
            summary_path = os.path.join(dataset_dir, "summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"Dataset Summary: {dataset_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Frames: {total_frames}\n")
                f.write(f"File Size: {file_size / (1024*1024):.2f} MB\n")
                f.write(f"Cameras: {', '.join(cameras)}\n")
                f.write(f"Data Columns: {len(frame_reader.frames.columns)}\n")
                f.write(
                    f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Images Exported: {images_exported}\n")

            print(f"Summary exported: {summary_path}")

            return {
                "dataset_name": dataset_name,
                "dataset_dir": dataset_dir,
                "csv": frame_data_path,
                "numpy": np_path,
                "columns": columns_path,
                "metadata": metadata_path,
                "summary": summary_path,
                "sample_images": sample_images_dir,
                "total_frames": total_frames,
                "images_exported": images_exported,
            }

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def export_sample_images(
        self, camera_reader, cameras, output_dir, total_frames, max_images_per_camera=10
    ):
        """
        Export sample images from all cameras.

        Args:
            camera_reader: CameraDataReader instance
            cameras: List of available cameras
            output_dir: Directory to save images
            total_frames: Total number of frames
            max_images_per_camera: Maximum images to save per camera

        Returns:
            int: Total number of images exported
        """

        total_exported = 0

        for camera in cameras:
            print(f"  Processing camera: {camera}")

            # Create camera subdirectory
            camera_dir = os.path.join(output_dir, camera)
            os.makedirs(camera_dir, exist_ok=True)

            # Try to export images from different frames
            exported_count = 0
            frame_step = max(1, total_frames // max_images_per_camera)

            for frame_idx in range(0, total_frames, frame_step):
                if exported_count >= max_images_per_camera:
                    break

                try:
                    # Get grayscale image
                    image = camera_reader.get_image(camera, frame_idx, "grayscale")

                    if image is not None:
                        # Save image
                        image_path = os.path.join(
                            camera_dir, f"frame_{frame_idx:04d}_grayscale.png"
                        )
                        Image.fromarray(image).save(image_path)
                        exported_count += 1

                        # Try to get semantic image if available
                        try:
                            semantic_image = camera_reader.get_image(
                                camera, frame_idx, "semantic"
                            )
                            if semantic_image is not None:
                                semantic_path = os.path.join(
                                    camera_dir, f"frame_{frame_idx:04d}_semantic.png"
                                )
                                Image.fromarray(semantic_image).save(semantic_path)
                        except Exception:
                            pass  # Semantic images might not be available

                except Exception:
                    # Skip frames with errors
                    continue

            print(f"    Camera {camera}: {exported_count} images exported")
            total_exported += exported_count

        return total_exported


def process_datasets(lac_files, output_dir="datasets", max_images=10):
    """
    Process multiple LAC datasets.

    Args:
        lac_files (list): List of paths to .lac files
        output_dir (str): Base output directory
    """

    print(f"Processing {len(lac_files)} LAC datasets")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    exporter = DatasetExporter(output_dir)
    results = []

    for i, lac_file in enumerate(lac_files, 1):
        print(f"\nDataset {i}/{len(lac_files)}")

        if not os.path.exists(lac_file):
            print(f"File not found: {lac_file}")
            continue

        result = exporter.export_dataset(lac_file, max_images)
        if result:
            results.append(result)
            print(f"Successfully processed: {result['dataset_name']}")
        else:
            print(f"Failed to process: {lac_file}")

    # Create consolidated summary
    create_consolidated_summary(results, output_dir)

    return results


def create_consolidated_summary(results, output_dir):
    """Create a summary of all processed datasets."""

    summary_path = os.path.join(output_dir, "consolidated_summary.json")

    consolidated_data = {
        "export_date": datetime.now().isoformat(),
        "total_datasets": len(results),
        "total_frames": sum(r["total_frames"] for r in results),
        "total_images": sum(r["images_exported"] for r in results),
        "datasets": [],
    }

    for result in results:
        consolidated_data["datasets"].append(
            {
                "name": result["dataset_name"],
                "frames": result["total_frames"],
                "images": result["images_exported"],
                "directory": result["dataset_dir"],
            }
        )

    with open(summary_path, "w") as f:
        json.dump(consolidated_data, f, indent=2)

    print(f"\nConsolidated summary created: {summary_path}")
    print(f"Total datasets processed: {len(results)}")
    print(f"Total frames: {consolidated_data['total_frames']}")
    print(f"Total images: {consolidated_data['total_images']}")


def main():
    """Main function with command line argument parsing."""

    parser = argparse.ArgumentParser(
        description="Explore and export multiple LAC datasets to organized folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single dataset
  python explore_dataset_flexible.py dataset1.lac
  
  # Process multiple datasets
  python explore_dataset_flexible.py dataset1.lac dataset2.lac dataset3.lac
  
  # Process all .lac files in current directory
  python explore_dataset_flexible.py *.lac
  
  # Specify custom output directory
  python explore_dataset_flexible.py *.lac --output my_datasets
  
  # Process files from a specific directory
  python explore_dataset_flexible.py /path/to/datasets/*.lac
        """,
    )

    parser.add_argument(
        "lac_files",
        nargs="+",
        help="LAC dataset files to process (can use glob patterns like *.lac)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="datasets",
        help="Output directory for exported datasets (default: datasets)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum sample images to export per camera (default: 10)",
    )

    args = parser.parse_args()

    # Expand glob patterns
    expanded_files = []
    for pattern in args.lac_files:
        if "*" in pattern or "?" in pattern:
            expanded_files.extend(glob.glob(pattern))
        else:
            expanded_files.append(pattern)

    # Remove duplicates and sort
    expanded_files = sorted(list(set(expanded_files)))

    if not expanded_files:
        print("No LAC files found matching the specified patterns")
        return

    print(f"Found {len(expanded_files)} LAC files to process:")
    for file in expanded_files:
        print(f"  {file}")

    # Process all datasets
    results = process_datasets(expanded_files, args.output, args.max_images)

    print("\nProcessing complete!")
    print(f"Check the '{args.output}/' directory for exported datasets")

    if results:
        print(f"\nSuccessfully processed {len(results)} datasets:")
        for result in results:
            print(
                f"  {result['dataset_name']}: {result['total_frames']} frames, {result['images_exported']} images"
            )


if __name__ == "__main__":
    main()
