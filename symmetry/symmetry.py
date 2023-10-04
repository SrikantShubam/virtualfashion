import os 
import sys
import dlib
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from standardize import extract_faces_no_save

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = predictor(img, face)
        nose = np.array([shape.part(33).x, shape.part(33).y])

        landmarks = [17, 26, 22, 21, 36, 45, 39, 42, 31, 35, 48, 54, 51, 57, 4, 14, 19, 24]
        mirror_mapping = {17: 26, 26: 17, 22: 21, 21: 22, 36: 45, 45: 36, 39: 42, 42: 39, 31: 35, 35: 31, 48: 54, 54: 48, 51: 57, 57: 51, 4: 14, 14: 4, 19: 24, 24: 19}

        ratios = []

        for idx in landmarks:
            point = np.array([shape.part(idx).x, shape.part(idx).y])
            mirror_idx = mirror_mapping[idx]
            mirror_point = np.array([shape.part(mirror_idx).x, shape.part(mirror_idx).y])

            if display_image:
                cv2.line(img_rgb, tuple(nose), tuple(point), (0, 255, 0), 1)
                cv2.circle(img_rgb, tuple(point), 2, (0, 255, 255), -1)

            ratio = euclidean_distance(nose, point) / euclidean_distance(nose, mirror_point)
            ratios.append(ratio)

        symmetry_score = np.mean(ratios)

        if display_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()

        return symmetry_score

def facial_symmetry(input_img, predictor_path, display_image=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Check if input_img is a string (path) or numpy array (loaded image)
    if isinstance(input_img, str):
        img = cv2.imread(input_img)
    else:
        img = input_img
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = predictor(img, face)
        nose = np.array([shape.part(33).x, shape.part(33).y])

        landmarks = [17, 26, 22, 21, 36, 45, 39, 42, 31, 35, 48, 54, 51, 57, 4, 14, 19, 24]
        mirror_mapping = {17: 26, 26: 17, 22: 21, 21: 22, 36: 45, 45: 36, 39: 42, 42: 39, 31: 35, 35: 31, 48: 54, 54: 48, 51: 57, 57: 51, 4: 14, 14: 4, 19: 24, 24: 19}

        ratios = []

        for idx in landmarks:
            point = np.array([shape.part(idx).x, shape.part(idx).y])
            mirror_idx = mirror_mapping[idx]
            mirror_point = np.array([shape.part(mirror_idx).x, shape.part(mirror_idx).y])

            if display_image:
                cv2.line(img_rgb, tuple(nose), tuple(point), (0, 255, 0), 1)
                cv2.circle(img_rgb, tuple(point), 2, (0, 255, 255), -1)

            ratio = euclidean_distance(nose, point) / euclidean_distance(nose, mirror_point)
            ratios.append(ratio)

        symmetry_score = np.mean(ratios)

        if display_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()

        return symmetry_score







# Upload image and model file
image_filename ="../testcases/largeheight.jpg"
predictor_filename = "../dependencies/shape_predictor_68_face_landmarks.dat"  # Assuming you've uploaded this too
faces = extract_faces_no_save(image_filename)

# Assuming you want to process the first detected face
if faces:
    symmetry_ratio = facial_symmetry(faces[0], predictor_filename,display_image=False)
    print(f"Facial Symmetry Ratio: {symmetry_ratio}")
else:
    print("No face detected.")
