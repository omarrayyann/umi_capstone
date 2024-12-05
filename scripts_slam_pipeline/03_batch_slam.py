"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask

def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs
        )
    except subprocess.TimeoutExpired as e:
        return e

@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="Don't pull docker image from Docker Hub")
@click.option('-s', '--setting_file', default="gopro12.yaml", help='Path to custom settings YAML file')
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull, setting_file):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file(), f"Map file {map_path} does not exist."

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    # Pull Docker image if required
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

    # Resolve the settings file path
    if setting_file is not None:
        setting_file = pathlib.Path(os.path.expanduser(setting_file)).absolute()
        assert setting_file.is_file(), f"Settings file {setting_file} does not exist."
        settings_mount_target = pathlib.Path('/settings').joinpath(setting_file.name)
    else:
        # Use default settings file inside the Docker container
        settings_mount_target = pathlib.Path('/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml')

    # Prepare volume mounts that are common to all runs
    map_mount_source = map_path
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
    volume_mounts_base = [
        '--volume', f"{map_mount_source.parent}:{map_mount_target.parent}",
    ]

    if setting_file is not None:
        volume_mounts_base.extend([
            '--volume', f"{setting_file.parent}:{settings_mount_target.parent}"
        ])

    with tqdm(total=len(input_video_dirs)) as pbar:
        # One chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in input_video_dirs:
                video_dir = video_dir.absolute()
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    pbar.update(1)
                    continue

                # Paths inside the Docker container
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')

                # Find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple

                # Create SLAM mask
                slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
                slam_mask = draw_predefined_mask(
                    slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                # Build volume mounts for this video_dir
                volume_mounts = [
                    '--volume', f"{video_dir}:{'/data'}",
                ] + volume_mounts_base

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
                    '--load_map', str(map_mount_target),
                    '--mask_img', str(mask_path),
                    '--max_lost_frames', str(max_lost_frames)
                ]

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # Limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                futures.add(executor.submit(
                    runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout
                ))

            # Wait for all futures to complete
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Done! Results:")
    for future in completed:
        print(future.result())

if __name__ == "__main__":
    main()
