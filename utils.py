from fastapi import UploadFile
import numpy as np
import cv2

def convert_image_to_np_array(img: UploadFile) -> np.ndarray | None: 
    contents = img.file.read()
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return decoded
