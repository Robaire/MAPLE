#!/usr/bin/env python

from lac_data import PlaybackAgent
import numpy as np
import cv2
import carla
import torch
from math import hypot
from pytransform3d.transformations import concat, invert_transform
from fastsam import FastSAM, FastSAMPrompt
import os
from maple.boulder import BoulderDetector
from collections import defaultdict

def transform_points(points_xyz, transform):
    """
    Apply a 4x4 transformation to a list or array of 3D points,
    with detailed debugging output if the input isn't as expected.
    """
    print("\n[transform_points] Starting transformation.")
    print(f"Original input type: {type(points_xyz)}")

    points_xyz = np.asarray(points_xyz)
    print(
        f"Converted to np.ndarray with shape: {points_xyz.shape}, dtype: {points_xyz.dtype}"
    )
    # Defensive checks
    if points_xyz is None:
        print("[transform_points] Warning: points_xyz is None")
        return np.empty((0, 3))

    points_xyz = np.asarray(points_xyz)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        print(f"[transform_points] Invalid shape: {points_xyz.shape}")
        return np.empty((0, 3))
    # Final check
    if points_xyz.shape[1] != 3:
        raise ValueError(
            f"[transform_points] After processing, points must have shape (N,3). Got {points_xyz.shape}."
        )

    # Continue with transformation
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    points_homogeneous = np.hstack((points_xyz, ones))  # (N, 4)

    # print(f"[transform_points] Built homogeneous points with shape: {points_homogeneous.shape}")

    points_transformed_homogeneous = (transform @ points_homogeneous.T).T  # (N, 4)
    points_transformed = points_transformed_homogeneous[:, :3]

    # print(f"[transform_points] Finished transformation. Output shape: {points_transformed.shape}\n")

    return points_transformed

def finalize(agent):
    min_det_threshold = 2

    if agent.frame > 15000:
        min_det_threshold = 3

    if agent.frame > 35000:
        min_det_threshold = 5
    # TODO: CREATE FICTITIOUS MAP
    g_map = agent.g_map
    gt_map_array = g_map


    N = gt_map_array.shape[
        0
    ]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
    x_min, y_min = gt_map_array[0][0][0], gt_map_array[0][0][0]
    resolution = 0.15

    # Calculate indices for center 2x2m region
    center_x_min_idx = int(round((-1 - x_min) / resolution))  # -.5m in x
    center_x_max_idx = int(round((1 - x_min) / resolution))  # +.5m in x
    center_y_min_idx = int(round((-1 - y_min) / resolution))  # -.5m in y
    center_y_max_idx = int(round((1 - y_min) / resolution))  # +.5m in y

    # # setting all rock locations to 0
    # for i in range(agent.map_length_testing):
    #     for j in range(agent.map_length_testing):
    #         agent.g_map_testing.set_cell_rock(i, j, 0)

    clusters = defaultdict(list)
    filtered_detections = []

    # First pass: create clusters
    for x_rock, y_rock in agent.all_boulder_detections:
        # Convert to grid coordinates
        i = int(round((x_rock - x_min) / resolution))
        j = int(round((y_rock - y_min) / resolution))

        # Create cluster key based on grid cell
        cluster_key = (i, j)
        clusters[cluster_key].append([x_rock, y_rock])

    final_clusters = []

    # Second pass: process clusters and filter outliers
    for (i, j), detections in clusters.items():
        # Skip clusters with less than 2 detections
        if len(detections) < min_det_threshold:
            continue

        final_clusters.extend(clusters[(i, j)])

        # Skip if in center region
        if (
            center_x_min_idx <= i <= center_x_max_idx
            and center_y_min_idx <= j <= center_y_max_idx
        ):
            continue

        # Sanity check: make sure we are within bounds
        if 0 <= i < N and 0 <= j < N:
            # Calculate cluster center
            x_center = float(np.mean([x for x, y in detections]))
            y_center = float(np.mean([y for x, y in detections]))

            # Convert back to grid coordinates for the map
            i_center = int(round((x_center - x_min) / resolution))
            j_center = int(round((y_center - y_min) / resolution))

            # Set rock location at cluster center
            agent.g_map[i_center, j_center] = 1

            # Store the cluster center as a simple list
            filtered_detections.append([x_center, y_center])

if __name__ == "__main__":
    # Initialize the playback agent
    agent = PlaybackAgent("resources/beaver_6.lac")
    output_dir = "boulder_maps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Starting playback...")

    frame = 1
    done = False

    # Initialize the BoulderDetector
    agent.detectorFront = BoulderDetector(agent, carla.sensorPosition.FrontLeft, carla.SensorPosition.FrontRight)
    agent.detectorBack = BoulderDetector(
            agent, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )
    agent.all_boulder_detections = []
    agent.large_boulder_detections = [(0,0,2.5)]
    agent.rock_map_testing = np.zeros((180,180))

    while not done:
        # Get input data from the cameras
        estimate = agent.get_transform()
        input_data = agent.input_data()
        # Check if we reached the end of the recording
        done = agent.at_end()
        
        # Get camera image
        left_camera = "FrontLeft"
        if agent.USE_BACK_CAM and not agent.USE_FRONT_CAM:
            estimate = estimate
            correction_T = agent.T_world_correction_back
        elif agent.USE_FRONT_CAM and not agent.USE_BACK_CAM:
            estimate = estimate
            correction_T = agent.T_world_correction_front

        try:
            # Check if camera data exists in input_data
            if "Grayscale" not in input_data:
                print(f"Frame {int(frame)}: No Grayscale data found in input_data")
                frame = agent.step_frame()
                continue
                
            # Get camera image
            left_image = input_data["Grayscale"].get(left_camera)
            
            # Check if image is valid
            if left_image is None:
                print(f"Frame {int(frame)}: Missing camera image")
                frame = agent.step_frame()
                continue
                
            # Process frames every 10 frames to reduce processing time
            if frame % 10 == 0:
                print(f"Processing frame {int(frame)}...")
                # run the BoulderDetector on the image
                detections, ground_points = agent.detectorFront(input_data)

                large_boulders_detections = agent.detectorFront.get_large_boulders()

                detections_back, _ = agent.detectorBack(input_data)

                # Get all detections in the world frame
                rover_world = estimate
                boulders_world = [
                    concat(boulder_rover, rover_world) for boulder_rover in detections
                ]

                ground_points_world = [
                    concat(ground_point, rover_world) for ground_point in ground_points
                ]

                boulders_world_back = [
                    concat(boulder_rover, rover_world) for boulder_rover in detections_back
                ]

                large_boulders_detections = [
                    concat(boulder_rover, rover_world)
                    for boulder_rover in large_boulders_detections
                ]

                large_boulders_xyr = [
                    (b_w[0, 3], b_w[1, 3], 0.3) for b_w in large_boulders_detections
                ]

                nearby_large_boulders = []
                for large_boulder in large_boulders_xyr:
                    print("large boulder: ", large_boulder)
                    (bx, by, _) = large_boulder  # assuming large_boulder is (x, y)

                    distance = hypot(bx - estimate[0,3], by - estimate[1,3])

                    if distance <= 2.0:
                        nearby_large_boulders.append(large_boulder)
                print("large boulders ", nearby_large_boulders)

                # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
                if len(nearby_large_boulders) > 0:
                    #self.navigator.add_large_boulder_detection(nearby_large_boulders)
                    agent.large_boulder_detections.extend(nearby_large_boulders)

                # If you just want X, Y coordinates as a tuple
                boulders_xyz = [(b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world]
                boulders_xyz_back = [
                    (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in boulders_world_back
                ]

                ground_points_xyz = [
                    (b_w[0, 3], b_w[1, 3], b_w[2, 3]) for b_w in ground_points_world
                ]

                # print("boulders detected in front: ", len(boulders_xyz))
                # print("boulders detected in back: ", len(boulders_xyz_back))
                correction_T = agent.get_initial_position() @ np.linalg.inv(
                estimate)

                if len(boulders_xyz) > 0:
                    boulders_world_corrected = transform_points(boulders_xyz, correction_T)
                    agent.all_boulder_detections.extend(boulders_world_corrected[:, :2])
                    # print("len(boulders)", len(self.all_boulder_detections))

                if len(boulders_xyz_back) > 0:
                    boulders_world_back_corrected = transform_points(
                        boulders_xyz_back, correction_T
                    )
                    agent.all_boulder_detections.extend(boulders_world_back_corrected[:, :2])
            
        except Exception as e:
            import traceback
            print(f"Frame {int(frame)}: Error processing frame: {str(e)}")
            traceback.print_exc()
        
        # Step the agent to the next frame
        frame = agent.step_frame()