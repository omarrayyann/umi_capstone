import cv2
import subprocess
import re

def list_v4l2_devices():
    """
    Lists video devices using `v4l2-ctl --list-devices`.
    
    Returns:
        dict: A dictionary where keys are device descriptions and values are lists of device paths.
    """
    devices = {}
    ls_output = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
    lines = ls_output.stdout.splitlines()
    
    current_device = None
    for line in lines:
        if not line.startswith('\t'):
            # This is a device description line
            current_device = line.strip(': ')
            devices[current_device] = []
        else:
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

def get_image_transform(in_res, out_res, crop_ratio=0.65):
    """
    Configures image transformation parameters.

    Args:
        in_res (tuple): Input resolution (width, height).
        out_res (tuple): Output resolution (width, height).
        crop_ratio (float): Ratio for cropping the image.

    Returns:
        function: A function that applies the transformation to an input frame.
    """
    def transform(image):
        if image is None or image.size == 0:
            print("Warning: Empty frame received.")
            return None
        
        h, w = image.shape[:2]  # get actual dimensions from the frame
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # Ensure cropping region is within bounds
        center_x, center_y = w // 2, h // 2
        start_x = max(center_x - crop_w // 2, 0)
        end_x = min(center_x + crop_w // 2, w)
        start_y = max(center_y - crop_h // 2, 0)
        end_y = min(center_y + crop_h // 2, h)

        # Perform cropping and resizing
        cropped_img = image[start_y:end_y, start_x:end_x]
        
        # Check if the cropped image is empty before resizing
        if cropped_img.size == 0:
            print("Warning: Cropped image is empty.")
            return None

        return cv2.resize(cropped_img, out_res)

    return transform

if __name__ == "__main__":
    # Desired resolutions

    
    # List all video devices
    devices = list_v4l2_devices()
    
    # Find Elgato device paths
    elgato_paths = find_elgato_device(devices)
    
    if not elgato_paths:
        print("No Elgato device found.")
        exit(1)
    
    # Attempt to open each Elgato video path until successful
    cap = None
    for path in elgato_paths:
        print(f"Attempting to open {path}...")
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Successfully opened {path}")
            dev_video_path = path
            break
        else:
            print(f"Failed to open {path}")
            cap.release()
            cap = None
    
    if cap is None:
        print("Could not open any Elgato video devices.")
        exit(1)
    
    # Get image transform function
    img_tf = get_image_transform(in_res=capture_res, out_res=out_res)
    
    # Read and transform frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        transformed_frame = img_tf(frame)
        
        if transformed_frame is not None:
            cv2.imshow('Transformed Frame', transformed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
