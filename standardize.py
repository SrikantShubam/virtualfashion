import dlib
import cv2
import os

def extract_faces(image_path, output_dir):
    # Load Dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Read the image
    img = cv2.imread(image_path)

    # Detect faces
    faces = detector(img)

    # Loop through detected faces and save them
    for i, d in enumerate(faces):
        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        face = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_dir, f'face_{i}.jpg'), face)

# # Test
# extract_faces("/content/man_largebackground.jpg", "/content/extract")

def extract_faces_no_save(image_path):
    """
    Extracts faces from an image.

    :param image_path: Path to the input image.
    :return: List of cropped face images.
    """
    # Load Dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Read the image
    img = cv2.imread(image_path)

    # Detect faces
    faces = detector(img)

    # Create a list to store the cropped face images
    cropped_faces = []

    # Loop through detected faces and add them to the cropped_faces list
    for i, d in enumerate(faces):
        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        face = img[y:y+h, x:x+w]
        cropped_faces.append(face)

    return cropped_faces

# # Test
# faces = extract_faces("/path/to/your/image.jpg")
# for i, face in enumerate(faces):
#     cv2.imshow(f'Face {i}', face)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
