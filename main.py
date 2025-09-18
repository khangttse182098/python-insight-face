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
from utils import classify_pose, compute_pose, convert_image_to_np_array
from schema import face_schema

load_dotenv()

# env
face_collection = os.getenv("face_embedding", "placeholder")

# milvus client
milvusClient = MilvusClient("./milvus.db")
if not milvusClient.has_collection(face_collection):
    milvusClient.create_collection(
        collection_name=face_collection, schema=face_schema, dimension=512
    )

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


@app.post("/register")
async def register(res: Response, userId: str = Form(...), img: UploadFile = Form(...)):
    # Check if user already has 4 images
    existed_face_list = (
        milvusClient.query(
            collection_name=face_collection,
            filter=f'code == "{userId}"',
            output_fields=["vector", "pose"],
        ),
    )

    # unwrap the tuple
    if isinstance(existed_face_list, tuple):
        existed_face_list = existed_face_list[0]

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
        if e["pose"] == new_pose.value:
            return JSONResponse(
                status_code=400,
                content={"message": "Tư thế đã tồn tại!"},
            )

    # Add the pose to database if not exist
    new_record = {
        "code": userId,
        "pose": new_pose.value,
        "vector": new_face.normed_embedding,
    }
    milvusClient.insert(collection_name=face_collection, data=new_record)
    return JSONResponse(
        status_code=200,
        content={
            "message": "Thêm tư thế mới thành công!",
            "data": {"pose": new_pose.value},
        },
    )


@app.post("/verify")
async def verify_person(
    res: Response, userId: str = Form(...), comparedImg: UploadFile = Form(...)
):
    # get user face vectors
    face_vector_list = (
        milvusClient.query(
            collection_name=face_collection,
            filter=f'code == "{userId}"',
            output_fields=["vector"],
        ),
    )

    # unwrap the tuple
    if isinstance(face_vector_list, tuple):
        face_vector_list = face_vector_list[0]

    # convert to array
    decodedImg = convert_image_to_np_array(comparedImg)
    compared_face = model.get(decodedImg)

    if len(compared_face) > 0:
        compared_face_embedding = compared_face[0].normed_embedding

        # Compute similarity (cosine distance) to all target embeddings
        similarities = [
            np.dot(compared_face_embedding, e["vector"]) for e in face_vector_list
        ]

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
