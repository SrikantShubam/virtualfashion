# # import cv2
# # import os

# # face_detector = cv2.CascadeClassifier('./dependencies/haarcascade_frontalface_default.xml')
# # pics = os.listdir('test cases')

# # # Specify the desired width and height for the display window
# # window_width = 800
# # window_height = 600

# # cv2.namedWindow("Image Display", cv2.WINDOW_NORMAL)  # Create a resizable window

# # for pic in pics:
# #     image_path = os.path.join('test cases', pic)
# #     image = cv2.imread(image_path)
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #     # Detect faces
# #     faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
# #     if image is not None:
# #         for (x, y, w, h) in faces:
# #             # Draw a rectangle around each detected face
# #             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         cv2.resizeWindow("Image Display", window_width, window_height)   
# #         cv2.imshow("Image Display", image)
# #          # Set the window size
# #         cv2.waitKey(0)
# #     else:
# #         print(f"Error loading image: {image_path}")

# # cv2.destroyAllWindows()

# # import cv2
# # window_width = 800
# # window_height = 600
# # # Load the Haar Cascade classifier for face detection
# # # face_cascade = cv2.CascadeClassifier('dependencies\haarcascade_frontalface_default.xml')

# # # Load an image
# # image_path = 'testcases\largeheight.jpg'
# # image = cv2.imread(image_path)
# # # cv2.resizeWindow("Image Display", window_width, window_height)
# # cv2.imshow("img",image)
# # cv2.waitKey(0)



# import cv2

# # Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load an image
# image_path = 'testcases\largeheight.jpg'
# image = cv2.imread(image_path)

# # Convert the image to grayscale (Haar Cascade works on grayscale images)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detect faces in the grayscale image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# # Create a named window with a specific size
# cv2.namedWindow('Detected Faces', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing

# # Set the window size to 800x600 pixels
# cv2.resizeWindow('Detected Faces', 800, 600)

# # Draw rectangles around the detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Display the image with detected faces
# cv2.imshow('Detected Faces', image)

# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-----------attempt 2  retinaface-----------------
import cv2
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
image_path = 'testcases\largeheight.jpg'
image = cv2.imread(image_path)
resp = RetinaFace.detect_faces(image_path)
# print(resp['face_1']['facial_area'])
print(resp)
# detected_info = {
#     'face_1': {
#         'score': 0.999767005443573,
#         'facial_area': [1308, 1994, 2217, 3277],
#         'all_landmarks': [
#             [1552.5781, 2511.863],
#             [1980.2604, 2483.8267],
#             [1786.8983, 2726.426],
#             [1576.0729, 2887.169],
#             [2015.0969, 2864.732]
#         ]
#     }
# }

# # Plot the image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# # Iterate through the detected faces
# for face_name, face_info in detected_info.items():
#     # Get facial area coordinates
#     x1, y1, x2, y2 = face_info['facial_area']

#     # Draw a rectangle around the facial area
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Iterate through all detected landmarks
#     for landmark in face_info['all_landmarks']:
#         # Get landmark coordinates
#         landmark_x, landmark_y = landmark

#         # Draw a point for each landmark
#         cv2.circle(image, (int(landmark_x), int(landmark_y)), 80, (255, 0, 0), -1)

# # Display the image with rectangles and landmarks
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # Turn off axis labels
# plt.show()
# import matplotlib.pyplot as plt
# faces = RetinaFace.extract_faces(img_path = image_path, align = True)
# all_faces=[]
# for face in faces:
#     # Append the face to the list of all faces
#     all_faces.append(face)


# output_dir = 'saved_faces'
# os.makedirs(output_dir, exist_ok=True)

# # Loop through the faces and save each one
# for i, face in enumerate(all_faces):
#     # Generate a unique filename for each face
#     face_filename = os.path.join(output_dir, f'extractedf_{i}.png')
    
#     # Save the face image
#     cv2.imwrite(face_filename, face[:,:,::-1])



# x1, y1, x2, y2 = [1308, 1994, 2217, 3277]
# # # Create a named window with a specific size

# # Draw a rectangle on the image
# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

# cv2.namedWindow('Image with Rectangle', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing

# # Set the window size to 800x600 pixels
# cv2.resizeWindow('Image with Rectangle', 800, 600)



# cv2.imshow('Image with Rectangle', image)
# cv2.waitKey(0)