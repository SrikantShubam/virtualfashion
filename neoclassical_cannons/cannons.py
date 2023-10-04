import cv2
import dlib
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from standardize import extract_faces_no_save
def neoclassical_canon_ratios(img):
    # Initialize dlib's face detector and the facial landmarks predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../dependencies/shape_predictor_68_face_landmarks.dat")

    # Euclidean distance function
    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    # Read and convert image to grayscale
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return None

    # Get landmarks for the first detected face
    landmarks = predictor(gray, faces[0])

    # Extract measurements
    facetop_eyebrows = euclidean_distance((landmarks.part(19).x, landmarks.part(19).y), (landmarks.part(24).x, landmarks.part(24).y))
    eyebrows_nose = euclidean_distance((landmarks.part(21).x, landmarks.part(21).y), (landmarks.part(22).x, landmarks.part(22).y))
    nose_jaw = euclidean_distance((landmarks.part(27).x, landmarks.part(27).y), (landmarks.part(8).x, landmarks.part(8).y))
    interocular = euclidean_distance((landmarks.part(39).x, landmarks.part(39).y), (landmarks.part(42).x, landmarks.part(42).y))
    nose_width = euclidean_distance((landmarks.part(31).x, landmarks.part(31).y), (landmarks.part(35).x, landmarks.part(35).y))
    pupil_outer_right = euclidean_distance((landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y))
    pupil_outer_left = euclidean_distance((landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y))
    right_eye_width = euclidean_distance((landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y))
    left_eye_width = euclidean_distance((landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y))
    mouth_width = euclidean_distance((landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y))
    face_width = euclidean_distance((landmarks.part(0).x, landmarks.part(0).y), (landmarks.part(16).x, landmarks.part(16).y))

    # Calculate neoclassical canon ratios
    ratios = {
        'facetop-eyebrows/eyebrows-nose': facetop_eyebrows / eyebrows_nose,
        'eyebrows-nose/nose-jaw': eyebrows_nose / nose_jaw,
        'facetop-eyebrows/nose-jaw': facetop_eyebrows / nose_jaw,
        'interocular/nose width': interocular / nose_width,
        'interocular/pupil-outer eye(right)': interocular / pupil_outer_right,
        'interocular/pupil-outer eye(left)': interocular / pupil_outer_left,
        'right eye width/left eye width': right_eye_width / left_eye_width,
        'mouth width/(1.5 x nose width)': mouth_width / (1.5 * nose_width),
        'face width/(4x nose width)': face_width / (4 * nose_width)
    }

    return ratios

# Example usage
img_path = "../testcases/largeheight.jpg"
extracted=extract_faces_no_save(img_path)

if extracted:
    cannons = neoclassical_canon_ratios(extracted[0])
    print(cannons)
else:
    print("No face detected.")

