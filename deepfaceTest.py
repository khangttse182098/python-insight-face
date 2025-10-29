from deepface import DeepFace
import cv2

img_path = "test_khang_phone_3.jpg"
faces = DeepFace.extract_faces(img_path=img_path, anti_spoofing=True)

img = cv2.imread(img_path)
img_h, img_w = img.shape[:2]

print(faces)
for f in faces:
    box = f["facial_area"]  # {'x':..., 'y':..., 'w':..., 'h':...}
    face_ratio = box["w"] / img_w  # % width of face vs image

    if face_ratio > 0.45:  # if face covers >45% of image, reject
        print("❌ Fake detected: face too close!")
    else:
        print("✅ Possibly real:", f["is_real"])
