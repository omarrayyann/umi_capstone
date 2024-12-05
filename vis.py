import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # Check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width

def plot_trajectory_with_arrow(ax, x_values, y_values, z_values, current_idx, window_size=5):
    """Plot the 3D trajectory with a direction arrow."""
    ax.clear()
    
    # Set fixed limits for a consistent scale
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Plot the full trajectory
    ax.plot(x_values[:current_idx + 1], y_values[:current_idx + 1], z_values[:current_idx + 1], color='blue')

    # Draw an arrow indicating the current direction of motion
    if current_idx > window_size:
        # Calculate direction vector over the window
        start_idx = max(0, current_idx - window_size)
        direction_vector = (
            x_values[current_idx] - x_values[start_idx],
            y_values[current_idx] - y_values[start_idx],
            z_values[current_idx] - z_values[start_idx]
        )
        ax.quiver(
            x_values[current_idx], y_values[current_idx], z_values[current_idx],
            direction_vector[0], direction_vector[1], direction_vector[2],
            color='red', length=0.1, normalize=True
        )

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory with Direction')

def synchronize_video_and_plot(csv_file, video_file, window_size=5):
    # Load trajectory data from CSV
    df = pd.read_csv(csv_file)
    timestamps = df['timestamp'].values
    x_values = df['x'].values
    y_values = df['y'].values
    z_values = df['z'].values

    # Load tag detection data from pickle
    with open('tag_detection.pkl', 'rb') as f:
        tag_detections = pickle.load(f)

    # Process tag detections to compute gripper widths
    gripper_timestamps = []
    gripper_widths = []

    for detection in tag_detections:
        timestamp = detection['time']
        tag_dict = detection['tag_dict']
        width = get_gripper_width(tag_dict, left_id=6, right_id=7, nominal_z=0.072, z_tolerance=0.008)
        gripper_timestamps.append(timestamp)
        gripper_widths.append(width)

    gripper_timestamps = np.array(gripper_timestamps)
    gripper_widths = np.array([w if w is not None else np.nan for w in gripper_widths])

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    window_width, window_height = 640, 360
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', window_width, window_height)

    # Initialize the 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    prev_gripper_width = None  # To track changes in gripper width

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the video frame
        frame = cv2.resize(frame, (window_width, window_height))

        # Get the current video timestamp
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        current_idx = (np.abs(timestamps - video_timestamp)).argmin()

        # Get the current gripper width
        gripper_idx = (np.abs(gripper_timestamps - video_timestamp)).argmin()
        current_gripper_width = gripper_widths[gripper_idx]

        # Print the current gripper width to the terminal
        if not np.isnan(current_gripper_width):
            if current_gripper_width != prev_gripper_width:
                print(f"Time {video_timestamp:.2f}s - Gripper Width: {current_gripper_width:.4f}")
                prev_gripper_width = current_gripper_width
        else:
            if prev_gripper_width is not None:
                print(f"Time {video_timestamp:.2f}s - Gripper Width: None")
                prev_gripper_width = None

        # Update the plot
        plot_trajectory_with_arrow(ax, x_values, y_values, z_values, current_idx, window_size)

        # Convert the matplotlib plot to an image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.resize(plot_img, (window_width, window_height))

        # Combine the video and plot side by side
        combined_frame = cv2.hconcat([frame, plot_img])
        cv2.imshow('Video with Trajectory', combined_frame)

        # Exit on 'q' key press
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_file = "camera_trajectory.csv"  # Replace with your CSV file path
    video_file = "raw_video.mp4"  # Replace with your video file path
    synchronize_video_and_plot(csv_file, video_file)

