from insightface.app import FaceAnalysis
import cv2
import numpy as np

# Load the face analysis model
model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Load images
multi_images = ['front.jpg', 'left.jpg', 'right.jpg', 'up.jpg'] 
multi_embeddings = []

for img_path in multi_images:
    img = cv2.imread(img_path)
    faces = model.get(img)
    if len(faces) > 0:
        embedding = faces[0].normed_embedding
        multi_embeddings.append(embedding)


# Load target images
img_new = cv2.imread("right2.jpg")
faces_new = model.get(img_new)

if len(faces_new) > 0:
    new_embedding = faces_new[0].normed_embedding
    
    # Compute similarity (cosine distance) to all target embeddings
    similarities = [np.dot(new_embedding, e) for e in multi_embeddings]
    
    # Take the maximum similarity
    max_similarity = max(similarities)
    print("Max similarity:", max_similarity)
    
    if max_similarity > 0.6:  # threshold (tune this)
        print("Target person recognized!")
    else:
        print("Unknown person")
