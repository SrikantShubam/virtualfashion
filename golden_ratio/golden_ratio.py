import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from standardize import extract_faces_no_save
def golden_ratio_analysis(img, display=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../dependencies/shape_predictor_68_face_landmarks.dat")

    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))
    
    def average_point(*points):
        x = sum(p[0] for p in points) // len(points)
        y = sum(p[1] for p in points) // len(points)
        return (x, y)


    # img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        interocular = euclidean_distance((landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(42).x, landmarks.part(42).y))
        nose_width = euclidean_distance((landmarks.part(31).x, landmarks.part(31).y), (landmarks.part(35).x, landmarks.part(35).y))
        under_eyes = euclidean_distance(average_point((landmarks.part(40).x, landmarks.part(40).y), (landmarks.part(41).x, landmarks.part(41).y)),
                                        average_point((landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(44).x, landmarks.part(44).y)))
        mouth_width = euclidean_distance((landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y))
        upper_lip_to_nose = euclidean_distance(average_point((landmarks.part(51).x, landmarks.part(51).y), (landmarks.part(52).x, landmarks.part(52).y), (landmarks.part(53).x, landmarks.part(53).y)),
                                              (landmarks.part(33).x, landmarks.part(33).y))
        upper_lip_to_jaw = euclidean_distance(average_point((landmarks.part(51).x, landmarks.part(51).y), (landmarks.part(52).x, landmarks.part(52).y), (landmarks.part(53).x, landmarks.part(53).y)),
                                            (landmarks.part(8).x, landmarks.part(8).y))
        lip_height = euclidean_distance((landmarks.part(51).x, landmarks.part(51).y), (landmarks.part(57).x, landmarks.part(57).y))
        nose_to_mouth = euclidean_distance((landmarks.part(33).x, landmarks.part(33).y), (landmarks.part(51).x, landmarks.part(51).y))
        eyebrow_distance = euclidean_distance((landmarks.part(21).x, landmarks.part(21).y), (landmarks.part(22).x, landmarks.part(22).y))
        face_width = euclidean_distance((landmarks.part(0).x, landmarks.part(0).y), (landmarks.part(16).x, landmarks.part(16).y))
        # Calculate ratios
        ratios = {
            'under_eyes/interocular': under_eyes / interocular,
            'under_eyes/nose_width': under_eyes / nose_width,
            'mouth_width/interocular': mouth_width / interocular,
            'upper_lip_to_nose/interocular': upper_lip_to_nose / interocular,
            'upper_lip_to_jaw/nose_width': upper_lip_to_jaw / nose_width,
            'interocular/lip_height': interocular / lip_height,
            'nose_width/interocular': nose_width / interocular,
            'nose_width/upper_lip_height': nose_width / upper_lip_to_nose,
            'interocular/nose_to_mouth': interocular / nose_to_mouth,
            'eyebrow_distance/face_width':eyebrow_distance / face_width,
        }
        
 

    
    aggregate_ratio = sum(ratios.values()) / len(ratios)

    if display:
        fig, ax = plt.subplots(1)
        fig, ax = plt.subplots(1, figsize=(15, 15))
        
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Define lines for each of your ratios
        lines = {
            "under_eyes": [(landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(42).x, landmarks.part(42).y)],
            "interocular": [(landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(42).x, landmarks.part(42).y)],
        
        }

        for name, coords in lines.items():
            ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], 'b-')
            midx = (coords[0][0] + coords[1][0]) / 2
            midy = (coords[0][1] + coords[1][1]) / 2
            ax.text(midx, midy, name, fontsize=9, color='red')
        
        # Add point markers for the start and end of each line
        for coord in set(sum(lines.values(), [])):
            ax.plot(coord[0], coord[1], 'ro')
       
        plt.show()

    return ratios, aggregate_ratio


image_filename ="../testcases/man_headup.jpg"

faces = extract_faces_no_save(image_filename)

# Assuming you want to process the first detected face
if faces:
    ratios,agg = golden_ratio_analysis(faces[0])
    print(f"Ratios: {ratios} AGG {agg}")
else:
    print("No face detected.")