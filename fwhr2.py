import cv2
import math
import mediapipe as mp
from google.colab.patches import cv2_imshow
# Function to calculate the angle between two lines
def calculate_angle(line1, line2):
    dx1 = line1[2] - line1[0]
    dy1 = line1[3] - line1[1]
    dx2 = line2[2] - line2[0]
    dy2 = line2[3] - line2[1]
    angle = math.atan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2)
    return math.degrees(angle)

# Load the image
image_path = '/content/img.jpeg'
image = cv2.imread(image_path)

# Face Landmark Detection using MediaPipe
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:

    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Convert the normalized coordinates of landmarks to pixel coordinates
        landmark_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks.landmark]

        # Initialize variables for right and left eye landmarks
        right_eye_coords = None
        left_eye_coords = None

        # Initialize variables for the topmost and lowest landmarks in the y-axis
        topmost_landmark = None
        lowest_landmark = None

        for i, point in enumerate(landmark_points):
            if i == 159 or i == 386:  # Right and left eye landmarks
                cv2.circle(image, point, 4, (0, 255, 0), -1)  # Draw green circles at eye landmarks
                if i == 159:
                    right_eye_coords = point
                else:
                    left_eye_coords = point
            else:
                cv2.circle(image, point, 4, (0, 0, 255), -1)  # Draw red circles at other landmarks

                # Find the topmost and lowest landmarks
                if topmost_landmark is None or point[1] < topmost_landmark[1]:
                    topmost_landmark = point
                if lowest_landmark is None or point[1] > lowest_landmark[1]:
                    lowest_landmark = point

        if right_eye_coords and left_eye_coords:
            # Calculate the midpoint of the line connecting the right and left eyes
            midpoint_x = (right_eye_coords[0] + left_eye_coords[0]) // 2
            midpoint_y = (right_eye_coords[1] + left_eye_coords[1]) // 2

            # Draw the midpoint in blue
            cv2.circle(image, (midpoint_x, midpoint_y), 4, (255, 0, 0), -1)  # Draw a blue circle at the midpoint

            # Draw a straight line between the green points (right and left eye landmarks)
            cv2.line(image, right_eye_coords, left_eye_coords, (0, 255, 0), 2)

            # Initialize a flag to check if the angle condition is met
            angle_condition_met = False

            # Try to meet the angle condition by changing the topmost and lowest landmarks
            while not angle_condition_met:
                # Calculate the angle between the line connecting the eyes and the line between topmost and lowest landmarks
                eye_line = [right_eye_coords[0], right_eye_coords[1], left_eye_coords[0], left_eye_coords[1]]
                top_low_line = [topmost_landmark[0], topmost_landmark[1], lowest_landmark[0], lowest_landmark[1]]
                angle = calculate_angle(eye_line, top_low_line)

                # Check if the angle is between 80 and 100 degrees
                if 70 <= angle <= 110:
                    angle_condition_met = True
                else:
                    # Change the topmost and lowest landmarks one at a time
                    if topmost_landmark[1] < lowest_landmark[1]:
                        topmost_landmark = (topmost_landmark[0], topmost_landmark[1] + 1)
                    else:
                        lowest_landmark = (lowest_landmark[0], lowest_landmark[1] + 1)

            # Draw a line between the eyes
            cv2.line(image, right_eye_coords, left_eye_coords, (255, 0, 0), 2)

            # Calculate the Euclidean distance between the topmost and lowest positions
            euclidean_distance = cv2.norm(topmost_landmark, lowest_landmark)

            # Print the Euclidean distance
            print(f"Euclidean Distance: {euclidean_distance}")

        # Color the topmost and lowest landmarks black
        cv2.circle(image, topmost_landmark, 4, (0, 0, 0), -1)
        cv2.circle(image, lowest_landmark, 4, (0, 0, 0), -1)

        # Draw a line between the topmost and lowest landmarks
        cv2.line(image, topmost_landmark, lowest_landmark, (0, 255, 0), 2)

        # Display the image with marked landmarks and lines
        cv2_imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No face detected in the image.")
