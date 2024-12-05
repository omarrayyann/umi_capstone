"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="Don't pull docker image from Docker Hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro not on gripper.")
@click.option('-s', '--setting_file', default="gopro12.yaml", help='Path to custom settings YAML file')
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask, setting_file):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file(), f"Missing {fn} in {video_dir}"

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull docker image if required
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')

    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    # Resolve the settings file path
    if setting_file is not None:
        setting_file = pathlib.Path(os.path.expanduser(setting_file)).absolute()
        assert setting_file.is_file(), f"Settings file {setting_file} does not exist."
    else:
        # Use default settings file inside the Docker container
        setting_file = pathlib.Path('/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml')

    # Prepare volume mounts
    volume_mounts = [
        '--volume', f"{video_dir}:{'/data'}",
        '--volume', f"{map_mount_source.parent}:{map_mount_target.parent}",
    ]

    # If a custom settings file is provided, mount its directory
    if setting_file.is_absolute() and setting_file.exists():
        settings_mount_target = pathlib.Path('/settings').joinpath(setting_file.name)
        volume_mounts.extend([
            '--volume', f"{setting_file.parent}:{settings_mount_target.parent}"
        ])
    else:
        settings_mount_target = setting_file  # This will use the default path inside Docker

    # Build the Docker command
    cmd = [
        'docker',
        'run',
        '--rm',
        *volume_mounts,
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', str(settings_mount_target),
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)


# %%
if __name__ == "__main__":
    main()
