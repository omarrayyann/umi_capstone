# eval_real_umi_simple.py
import pickle
import os
import time
import pathlib
import torch
import dill
import yaml
import json
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from omegaconf import OmegaConf
from multiprocessing.managers import SharedMemoryManager
import subprocess
import collections
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as R

import hydra
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from umi.real_world.bimanual_umi_env import BimanualUmiEnv  # Uncomment when environment is ready
from umi.real_world.real_inference_util import (
    get_real_obs_dict,
    get_real_umi_obs_dict,
    get_real_umi_action
)
from umi.common.pose_util import pose_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)

def list_v4l2_devices():
    """
    Lists video devices using `v4l2-ctl --list-devices`.

    Returns:
        dict: A dictionary where keys are device descriptions and values are lists of device paths.
    """
    devices = {}
    try:
        ls_output = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, check=True)
        lines = ls_output.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error listing V4L2 devices: {e}")
        return devices

    current_device = None
    for line in lines:
        if not line.startswith('\t') and line.strip() != '':
            # This is a device description line
            current_device = line.strip(': ')
            devices[current_device] = []
        elif line.startswith('\t') and current_device:
            # This is a device path line
            device_path = line.strip()
            devices[current_device].append(device_path)

    return devices


def find_elgato_device(devices):
    """
    Finds the Elgato Capture Card device paths.

    Args:
        devices (dict): Dictionary of device descriptions and their paths.

    Returns:
        list: List of Elgato device paths.
    """
    elgato_devices = []
    for description, paths in devices.items():
        if 'Elgato' in description:
            elgato_devices.extend(paths)
            print(f"Found Elgato device: {description} with paths {paths}")
    return elgato_devices


def load_normalizer(normalizer_path):
    """
    Load the LinearNormalizer from the given .pkl file using pickle.
    """
    try:
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        print("Loaded normalizer successfully.")
        return normalizer
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        return None


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--gopro_stream', '-g', required=True, help='GoPro Stream URL or device index (e.g., 0 for webcam)')
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency in seconds.")
@click.option('--verify_cameras', '-vc', is_flag=True, default=False, help='Verify and display connected cameras.')
@click.option('--normalizer', '-n', required=True, help='Path to normalizer .pkl file')
def main(input, output, normalizer, robot_config, gopro_stream, steps_per_inference, max_duration, frequency, command_latency, verify_cameras):
    import subprocess
    import collections

    print(normalizer)
    
    frame_queue = collections.deque(maxlen=2)
    pose_queue = collections.deque(maxlen=2)
    gripper_queue = collections.deque(maxlen=2)

    # Load robot config file
    try:
        robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    except FileNotFoundError:
        print(f"Robot configuration file not found: {robot_config}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing robot configuration file: {e}")
        return

    tx_left_right = np.array(robot_config_data.get('tx_left_right', [0, 0, 0]))
    tx_robot1_robot0 = tx_left_right
    robots_config = robot_config_data.get('robots', [])
    grippers_config = robot_config_data.get('grippers', [])

    if not robots_config:
        print("No robot configurations found in the config file.")
        return
    
    normalizer = load_normalizer(normalizer)
    if normalizer is None:
        print("Failed to load normalizer. Exiting.")
        return

    # Load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
        return

    try:
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    cfg = payload.get('cfg')
    if not cfg:
        print("Configuration 'cfg' not found in the checkpoint.")
        return

    print("Model Name:", cfg.policy.obs_encoder.model_name)
    print("Dataset Path:", cfg.task.dataset.dataset_path)

    # Setup experiment
    dt = 1 / frequency

    # Optional: Verify and display connected cameras
    if verify_cameras:
        print("Scanning for connected cameras...")
        devices = list_v4l2_devices()
        elgato_paths = find_elgato_device(devices)

        if not elgato_paths:
            print("No Elgato devices found.")
            # Optionally, list all available cameras
            available_cameras = devices
            if available_cameras:
                print("Available cameras:")
                for desc, paths in available_cameras.items():
                    print(f"{desc}: {paths}")
            else:
                print("No video devices found.")
            return

        # Attempt to open each Elgato video path until successful
        cap_verify = None
        for path in elgato_paths:
            print(f"Attempting to open {path} for verification...")
            cap_verify = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap_verify.isOpened():
                print(f"Successfully opened {path} for verification.")
                # Display the camera feed until 'q' is pressed
                while True:
                    ret, frame = cap_verify.read()
                    if not ret:
                        print(f"Failed to read frame from {path}.")
                        break
                    cv2.imshow(f'Verify {path}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap_verify.release()
                cv2.destroyAllWindows()
                break
            else:
                print(f"Failed to open {path} for verification.")
                cap_verify.release()
                cap_verify = None

        if cap_verify is None:
            print("Could not open any Elgato video devices for verification.")
            return

    # Initialize video capture for GoPro stream or Elgato device
    devices = list_v4l2_devices()
    elgato_paths = find_elgato_device(devices)

    if elgato_paths:
        print("Elgato devices found. Attempting to use the first available Elgato device.")
        selected_path = elgato_paths[0]
        print(f"Selected Elgato device path: {selected_path}")
        cap = cv2.VideoCapture(selected_path, cv2.CAP_V4L2)
    else:
        # Fallback to using gopro_stream as device index or URL
        if gopro_stream.isdigit():
            gopro_stream = int(gopro_stream)
        cap = cv2.VideoCapture(gopro_stream)

    if not cap.isOpened():
        print(f"Failed to open GoPro stream or Elgato device: {gopro_stream}")
        return
    print("GoPro stream or Elgato device opened successfully.")

    # Define image transformation parameters
    # Adjust capture_res based on your camera's native resolution
    # For example, if Elgato captures at 1920x1080, set accordingly
    capture_res = (640,480)  # Example input resolution (width, height)
    out_res = (224, 224)         # Desired output resolution
    img_tf = get_image_transform(input_res=capture_res, output_res=out_res)


    # Load model
    try:
        cls = hydra.utils.get_class(cfg._target_)
    except Exception as e:
        print(f"Error retrieving class '{cfg._target_}': {e}")
        cap.release()
        return

    try:
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    except Exception as e:
        print(f"Error loading workspace payload: {e}")
        cap.release()
        return

    policy = workspace.model
    if cfg.training.get('use_ema', False):
        policy = workspace.ema_model
    policy.num_inference_steps = 16  # DDIM inference iterations
    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    print("Warming up policy inference")

    shm = shared_memory.SharedMemory(name='my_pose')

    # Initialize episode_start_pose based on initial observations
    # Since we don't have actual observations, we'll initialize with zeros
    episode_start_pose = [np.zeros(6, dtype=np.float32) for _ in robots_config]  # [x, y, z, roll, pitch, yaw]


    # Initialize mock robot observations
    # Assuming only one robot; extend if multiple robots are configured
    mock_robot_obs = {}
    for robot_id in range(len(robots_config)):
        mock_robot_obs[f'robot{robot_id}_eef_pos'] = np.random.rand(2, 3).astype(np.float32)  # [x, y, z]
        mock_robot_obs[f'robot{robot_id}_eef_rot_axis_angle'] = np.random.rand(2, 3).astype(np.float32)  # [roll, pitch, yaw]
        mock_robot_obs[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = np.random.rand(2, 6).astype(np.float32)  # [rot_6d representation]
        mock_robot_obs[f'robot{robot_id}_gripper_width'] = np.random.rand(2,1).astype(np.float32)  # [gripper width]


    # Initialize mock camera observation
    # If you prefer using actual camera frames, you can replace this with a real frame capture
    mock_camera_frame = np.random.rand(2, 224, 224, 3).astype(np.float32)  # Mock image
    mock_camera_frame = (mock_camera_frame * 255).astype(np.uint8)  # Convert to uint8

    # Prepare initial env_obs for warming up
    env_obs = {
        'camera0_rgb': mock_camera_frame,  # Shape: (1, 224, 224, 3)
        'robot0_gripper_width': np.random.rand(1,1).astype(np.float32),  # Random gripper width
    }
    env_obs.update(mock_robot_obs)  # Add robot observations
    
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs, 
            shape_meta=cfg.task.shape_meta, 
            obs_pose_repr=obs_pose_repr,
            tx_robot1_robot0=tx_robot1_robot0,
            episode_start_pose=episode_start_pose
        )
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)

        action = result['action_pred'][0].detach().cpu().numpy()
        
        assert action.shape[-1] == 10 * len(robots_config)
        action = get_real_umi_action(action, env_obs, action_pose_repr)

        assert action.shape[-1] == 7 * len(robots_config)
        del result
        # except Exception as e:
        #     print(f"Error during warming up: {e}")
        #     cap.release()
        #     return

    print('Ready!')

    try:
        # Start episode
        policy.reset()
        eval_t_start = time.time()

        iter_idx = 0

        start_time = time.time()
        start_pose  = np.ndarray((4, 4), dtype=np.float64, buffer=shm.buf)
        start_pose[3,3] = 1.0

        while True:
            t_cycle_end = iter_idx * dt

            # Capture frame from camera
            ret, frame = cap.read()
            print(frame.shape)
            if not ret:
                print("Failed to read frame from GoPro stream.")
                break
            
            end_time = time.time()
            print(1/(end_time-start_time))
            start_time = time.time()  

            # Create a NumPy array backed by shared memory
            pose = np.ndarray((4, 4), dtype=np.float64, buffer=shm.buf)
            gripper_width = pose[3,3]/600*0.085888858 + 0.04272118273535439


            pose[3,3] = 1

            gripper_queue.append(gripper_width)
            pose_queue.append(pose)
                
            transformed_frame = img_tf(frame)
            if transformed_frame is None:
                print("Transformed frame is None. Skipping this frame.")
                continue

            # Add the transformed frame to the queue
            frame_queue.append(transformed_frame)

            # Display the frame
            cv2.imshow('GoPro Stream', transformed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received. Exiting.")
                break
            
            # Ensure the queue has two frames
            if len(frame_queue) < 2 or len(pose_queue) < 2:
                print("Waiting for two frames to populate the queue.")
                continue

            # Convert deque to a list and stack frames along a new axis
            frames_list = list(frame_queue)
            obs_image = np.stack(frames_list, axis=0)  # Shape: (2, 224, 224, 3)

            # Normalize pixel values to [0, 1]
            obs_image = obs_image.astype(np.float32) / 255.0

            # # Prepare observation for the model
            # # Convert frame to [batch_size, channels, height, width] and normalize
            # obs_image = transformed_frame.astype(np.float32) / 255.0
            # obs_image = np.transpose(obs_image, (2, 0, 1))  # Convert from HWC to CHW
            # obs_image = np.expand_dims(obs_image, axis=0)   # Add batch dimension


            # Mock robot observations (replace with actual data if available)
            mock_robot_obs = {}
            for robot_id in range(len(robots_config)):

                # delta_pos1 = pose_queue[-2][:3,3] - pose_queue[-1][:3,3]
                # delta_pos2 = pose_queue[-3][:3,3] - pose_queue[-2][:3,3]
                last_two_delta_pos = np.array([pose_queue[-2][:3,3], pose_queue[-1][:3,3]])

                print(last_two_delta_pos)

                # rot_prev = R.from_matrix(pose_queue[-2][:3, :3])
                # rot_prev_prev = R.from_matrix(pose_queue[-3][:3, :3])
                # delta_rot2 = rot_prev.inv() * R.from_matrix(pose_queue[-1][:3, :3])
                # delta_rot1 = rot_prev_prev.inv() * rot_prev
                last_two_delta_rot = np.array([R.from_matrix(pose_queue[-2][:3, :3]).as_euler('xyz', degrees=False),
                                            R.from_matrix(pose_queue[-1][:3, :3]).as_euler('xyz', degrees=False)])
                

                rot_start = R.from_matrix(start_pose[:3, :3])
                delta_rot_wrt_start2 = rot_start.inv() * R.from_matrix(pose_queue[-1][:3, :3])
                delta_rot_wrt_start1 = rot_start.inv() * R.from_matrix(pose_queue[-2][:3, :3])
                last_two_delta_rot_wrt_start = np.array([
                    delta_rot_wrt_start1.as_euler('xyz', degrees=False),
                    delta_rot_wrt_start2.as_euler('xyz', degrees=False)
                ])

                last_two_grasping_width = np.array([gripper_queue[-2], gripper_queue[-1]]).reshape(2,1  )

                mock_robot_obs[f'robot{robot_id}_eef_pos'] = last_two_delta_pos  # [x, y, z]
                mock_robot_obs[f'robot{robot_id}_eef_rot_axis_angle'] = last_two_delta_rot  # [roll, pitch, yaw]
                mock_robot_obs[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = last_two_delta_rot_wrt_start  # [rot_6d representation]
                mock_robot_obs[f'robot{robot_id}_gripper_width'] = last_two_grasping_width

                print(mock_robot_obs[f'robot{robot_id}_eef_pos'])

            # # Populate env_obs
            # env_obs = {
            #     'camera0_rgb': obs_image,  # Shape: (224, 224, 3)
            # }
            mock_robot_obs['camera0_rgb'] = obs_image
            cv2.imshow('GoPro Stream', obs_image[0])

        
            # Run inference
            with torch.no_grad():
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=mock_robot_obs, 
                    shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_repr,
                    tx_robot1_robot0=None,
                    episode_start_pose=episode_start_pose
                )

                print(obs_dict_np)
                
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                # obs_dict = normalizer.normalize(obs_dict)

                result = policy.predict_action(obs_dict)
                raw_action = result['action_pred'][0].detach().cpu().numpy()
                # print("raw action: ", raw_action)
                action = get_real_umi_action(raw_action, env_obs, action_pose_repr)
                del result

            # Solve collisions
            target_pose = action.reshape(policy.num_inference_steps, 7)
            # for robot_idx in range(len(robots_config)):
            #     solve_table_collision(
            #         ee_pose=target_pose[robot_idx],
            #         gripper_width=target_pose[robot_idx][6],
            #         height_threshold=robots_config[robot_idx]['height_threshold']
            #     )
            # solve_sphere_collision(
            #     ee_poses=target_pose,
            #     robots_config=robots_config
            # )

            # print(target_pose.shape)

            # Prepare actions for environment
            action_env = np.zeros((7 * len(robots_config),))
            for robot_idx in range(len(robots_config)):
                action_env[7 * robot_idx: 7 * robot_idx + 6] = target_pose[robot_idx][:6]
                action_env[7 * robot_idx + 6] = target_pose[robot_idx][6]
            
                gripper_width_raw = target_pose[:,6][:]
                gripper_width =  (gripper_width_raw-0.04272118273535439)/0.085888858 * 600
                print("Gripper Width", gripper_width)
                print("Delta Pose", target_pose[:,0:3][:])

            # Execute actions

            # env.exec_actions(
            #     actions=[action_env],
            #     timestamps=[time.time() + command_latency],
            #     compensate_latency=True
            # )

            iter_idx += 1

            # print(action_env)

            # Check max duration
            if time.time() - eval_t_start > max_duration:
                print("Max Duration reached.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Session ended.")


if __name__ == '__main__':
    main()
