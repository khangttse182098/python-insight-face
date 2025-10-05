from typing import Optional
from pymilvus import MilvusClient
from fastapi import FastAPI, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
from dotenv import load_dotenv
import os

from type import Pose
from utils import classify_pose, compute_pose, convert_image_to_np_array
from schema import face_schema
from http import HTTPStatus

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
rec_model = FaceAnalysis(providers=["CPUExecutionProvider"])
rec_model.prepare(ctx_id=0, det_size=(640, 640))


@app.get("/reset/{userId}")
async def reset(userId: str):
    milvusClient.delete(collection_name=face_collection, filter=f'code == "{userId}"')
    return JSONResponse(
        status_code=HTTPStatus.OK, content={"message": "Delete success"}
    )


@app.get("/missing-pose/{userId}")
async def missing_pose(userId: str):
    # get existed faces
    existed_face_list = milvusClient.query(
        collection_name=face_collection,
        filter=f'code == "{userId}"',
        output_fields=["pose"],
    )

    # get all poses
    all_pose = {e.value for e in Pose}

    if len(existed_face_list):
        existed_pose_list = {e["pose"] for e in existed_face_list}

        # find missing pose
        missing_pose = all_pose - existed_pose_list

        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={"missingPose": list(missing_pose)},
        )
    else:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={"missingPose": list(all_pose)},
        )


@app.post("/register")
async def register(
    userId: str = Form(...),
    img: UploadFile = Form(...),
    checkedPose: Optional[str] = Form(None),
):
    # Check if user already has 4 images
    existed_face_list = milvusClient.query(
        collection_name=face_collection,
        filter=f'code == "{userId}"',
        output_fields=["vector", "pose"],
    )

    if len(existed_face_list) == len(Pose):
        return JSONResponse(
            status_code=HTTPStatus.CONFLICT, content={"message": "Đã có đủ hình ảnh!"}
        )

    # verify if there are faces
    decodedImg = convert_image_to_np_array(img)
    face_list = rec_model.get(decodedImg)

    # return if multiple faces detected
    if len(face_list) > 1:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"message": "Không thể có nhiều hơn 1 khuôn mặt!"},
        )

    # return if no face detected
    if len(face_list) == 0:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"message": "Không tìm thấy khuôn mặt"},
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
                status_code=HTTPStatus.BAD_REQUEST,
                content={"message": f"Tư thế {new_pose} đã tồn tại!"},
            )

    # Check if the pose is correctly pass
    if checkedPose is None:
        # Add the pose to database if not exist
        new_record = {
            "code": userId,
            "pose": new_pose.value,
            "vector": new_face.normed_embedding,
        }
        milvusClient.insert(collection_name=face_collection, data=new_record)

        # Return if success
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "message": "Thêm tư thế mới thành công!",
                "data": {"pose": new_pose.value},
            },
        )

    checkedPoseInt = int(checkedPose) if checkedPose is not None else None
    # If checkedPose doesn't match newPose
    if checkedPoseInt != new_pose.value:
        print(f"checkpose: {checkedPoseInt}, new_pose: {new_pose.value}")
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={
                "message": "Tư thế không khớp với yêu cầu!",
            },
        )

    # Add the pose to database if not exist
    new_record = {
        "code": userId,
        "pose": new_pose.value,
        "vector": new_face.normed_embedding,
    }
    milvusClient.insert(collection_name=face_collection, data=new_record)
    # Return if success
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "message": "Thêm tư thế mới thành công!",
            "data": {"pose": new_pose.value},
        },
    )


@app.post("/verify")
async def verify_person(userId: str = Form(...), comparedImg: UploadFile = Form(...)):
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
    compared_face = rec_model.get(decodedImg)

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
                status_code=HTTPStatus.OK, content={"message": "Khuôn mặt trùng khớp!"}
            )
        else:
            print("Unknown person")
            return JSONResponse(
                status_code=HTTPStatus.CONFLICT,
                content={"message": "Khuôn mặt không trùng khớp!"},
            )
    else:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"message": "Không tìm thấy khuôn mặt!"},
        )


@app.post("/pose")
async def checkPost(res: Response, img: UploadFile = Form(...)):
    # verify if there are faces
    decodedImg = convert_image_to_np_array(img)
    face_list = rec_model.get(decodedImg)

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
