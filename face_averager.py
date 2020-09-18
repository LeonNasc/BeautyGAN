#encoding:utf-8
import matplotlib.pyplot as plt

from facer import facer

# Load face images
path_to_images = "./FBP_male"
images = facer.load_images(path_to_images)

# Detect landmarks for each face
landmarks, faces = facer.detect_face_landmarks(images)

# Use  the detected landmarks to create an average face
average_face = facer.create_average_face(faces, landmarks, save_image=True)

# View the composite image
plt.imshow(average_face)
plt.show()
