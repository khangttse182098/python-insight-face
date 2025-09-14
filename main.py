from pandas.io.html import pprint_thing
from pymilvus import MilvusClient
from fastapi import FastAPI, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import cv2
import numpy as np

client = MilvusClient("./milvus.db")
if not client.has_collection("face_embedding"):
    client.create_collection(collection_name="face_embedding", dimension=512)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the face analysis model
model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

@app.post("/verify")
async def verify_person(res: Response, comparedImg: UploadFile = Form(...)):

    # Load images
    multi_images = ['front.jpg', 'left.jpg', 'right.jpg', 'up.jpg'] 
    multi_embeddings = []

    for img_path in multi_images:
        # prepare for model
        img = cv2.imread(img_path)
        faces = model.get(img)
        if len(faces) > 0:
            embedding = faces[0].normed_embedding
            multi_embeddings.append(embedding)


    # Load target images
    contents = comparedImg.file.read()
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    compared_face = model.get(img)

    if len(compared_face) > 0:
        compared_face_embedding = compared_face[0].normed_embedding
        
        # Add temp data
        # tmp_db_data = [{"id": k, "code": 10, "vector": v} for k, v in enumerate(multi_embeddings)]
        # db_res = client.insert(collection_name="face_embedding", data=tmp_db_data)

        # Get current user
        db_res = client.query(
            collection_name="face_embedding",
            filter="code == 10",
            output_fields=["vector"],
        )
        # print(db_res)

        # Compute similarity (cosine distance) to all target embeddings
        similarities = [np.dot(compared_face_embedding, e) for e in multi_embeddings]
        
        # Take the maximum similarity
        max_similarity = max(similarities)
        print("Max similarity:", max_similarity)
        
        if max_similarity > 0.6:  # threshold (tune this)
            print("Target person recognized!")
            return JSONResponse(status_code=200, content={"message": "Khuôn mặt trùng khớp!"})
        else:
            print("Unknown person")
            return JSONResponse(status_code=400, content={"message": "Khuôn mặt không trùng khớp!"})
    else:
        return JSONResponse(status_code=404, content={"message": "Không tìm thấy khuôn mặt!"})
