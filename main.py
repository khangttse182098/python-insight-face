from typing import List, cast
from pymilvus import MilvusClient
from fastapi import FastAPI, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from dotenv import load_dotenv
import os
from type import FaceRecord
from utils import classify_pose, compute_pose, convert_image_to_np_array

load_dotenv()

face_collection = os.getenv("face_embedding", "placeholder")

milvusClient = MilvusClient("./milvus.db")
if not milvusClient.has_collection(face_collection):
    milvusClient.create_collection(collection_name=face_collection, dimension=512)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the face analysis model
model = FaceAnalysis(providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))


@app.post("/pose")
async def checkPost(res: Response, img: UploadFile = Form(...)):
    # verify if there are faces
    decodedImg = convert_image_to_np_array(img)
    face_list = model.get(decodedImg)

    # return if multiple faces detected
    if len(face_list) > 1:
        return JSONResponse(
            status_code=400, content={"message": "Không thể có nhiều hơn 1 khuôn mặt!"}
        )

    # Get the pose
    face = face_list[0]
    landmarks = face.landmark_2d_106
    # show landmarks
    # scale = 0.3  # shrink to 30% size
    # resized = cv2.resize(decodedImg, None, fx=scale, fy=scale)
    #
    # for i, (x, y) in enumerate(landmarks):
    #     x, y = int(x * scale), int(y * scale)  # scale landmarks too
    #     cv2.putText(resized, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #
    # cv2.imshow("Landmarks", resized)
    # cv2.waitKey(0)
    yaw, pitch, roll = compute_pose(decodedImg, landmarks)
    pose = classify_pose(yaw, pitch, roll)
    print("Pose:", pose)

    return JSONResponse(status_code=200, content={"message": f"{pose}"})


@app.post("/register")
async def register(
    res: Response, userCode: str = Form(...), img: UploadFile = Form(...)
):
    # Check if user already has 4 images
    existed_face_list = cast(
        List[FaceRecord],
        milvusClient.query(
            collection_name=face_collection,
            filter=f"code == {userCode}",
            output_fields=["vector", "pose"],
        ),
    )

    if len(existed_face_list) == 4:
        return JSONResponse(status_code=409, content={"message": "Đã có đủ hình ảnh!"})

    # verify if there are faces
    decodedImg = convert_image_to_np_array(img)
    face_list = model.get(decodedImg)

    # return if multiple faces detected
    if len(face_list) > 1:
        return JSONResponse(
            status_code=400, content={"message": "Không thể có nhiều hơn 1 khuôn mặt!"}
        )

    # Get the pose
    new_face = face_list[0]
    landmarks = new_face.landmark_2d_106
    yaw, pitch, roll = compute_pose(decodedImg, landmarks)
    new_pose = classify_pose(yaw, pitch, roll)

    print("Pose:", new_pose)

    # Check if the pose already exist
    for e in existed_face_list:
        if e.pose == new_pose:
            return JSONResponse(
                status_code=400,
                content={"message": "Tư thế đã tồn tại!"},
            )

    # Add the pose to database if not exist
    # new_record: FaceRecord = {"id": }
    # milvusClient.insert(
    #     collection_name=face_collection,
    # )


@app.post("/verify")
async def verify_person(res: Response, comparedImg: UploadFile = Form(...)):

    # Load images
    multi_images = ["front.jpg", "left.jpg", "right.jpg", "up.jpg"]
    multi_embeddings = []

    for img_path in multi_images:
        # prepare for model
        img = cv2.imread(img_path)
        faces = model.get(img)
        if len(faces) > 0:
            embedding = faces[0].normed_embedding
            multi_embeddings.append(embedding)

    # convert to array
    decodedImg = convert_image_to_np_array(comparedImg)
    compared_face = model.get(decodedImg)

    if len(compared_face) > 0:
        compared_face_embedding = compared_face[0].normed_embedding

        # Add temp data
        # tmp_db_data = [{"id": k, "code": 10, "vector": v} for k, v in enumerate(multi_embeddings)]
        # db_res = milvusClient.insert(collection_name=face_collection, data=tmp_db_data)

        # Get current user
        db_res = milvusClient.query(
            collection_name=face_collection,
            filter="code == 10",
            output_fields=["vector"],
        )
        print(db_res)

        # Compute similarity (cosine distance) to all target embeddings
        similarities = [np.dot(compared_face_embedding, e) for e in multi_embeddings]

        # Take the maximum similarity
        max_similarity = max(similarities)
        print("Max similarity:", max_similarity)

        if max_similarity > 0.6:  # threshold (tune this)
            print("Target person recognized!")
            return JSONResponse(
                status_code=200, content={"message": "Khuôn mặt trùng khớp!"}
            )
        else:
            print("Unknown person")
            return JSONResponse(
                status_code=422, content={"message": "Khuôn mặt không trùng khớp!"}
            )
    else:
        return JSONResponse(
            status_code=404, content={"message": "Không tìm thấy khuôn mặt!"}
        )
