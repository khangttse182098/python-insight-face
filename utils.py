from fastapi import UploadFile
import numpy as np
import cv2

from type import Pose


def convert_image_to_np_array(img: UploadFile) -> np.ndarray:
    contents = img.file.read()
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Image could not be decoded")
    return decoded


def compute_pose(image, landmarks):
    # 2D landmark points (mapped for 106-point InsightFace model)
    image_points = np.array(
        [
            landmarks[80],  # Nose tip
            landmarks[0],  # Chin
            landmarks[93],  # Left eye corner
            landmarks[35],  # Right eye corner
            landmarks[61],  # Left mouth corner
            landmarks[52],  # Right mouth corner
        ],
        dtype=np.float32,
    )

    # 3D model reference points
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ],
        dtype=np.float32,
    )

    # Camera parameters
    h, w = image.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    dist_coeffs = np.zeros((4, 1))  # no distortion

    # SolvePnP to get rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rmat, tvec))

    # Decompose projection matrix to Euler angles
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)
    pitch, yaw, roll = eulerAngles.flatten()

    return yaw, pitch, roll


def classify_pose(yaw, pitch, roll, yaw_thresh=20, pitch_thresh=15):
    if yaw > yaw_thresh:
        return Pose.RIGHT
    elif yaw < -yaw_thresh:
        return Pose.LEFT
    elif pitch > pitch_thresh:
        return Pose.DOWN
    elif pitch < -pitch_thresh:
        return Pose.UP
    else:
        return Pose.FRONT
