import cv2
import dlib
import pygame
from pygame.locals import *

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize Pygame
pygame.init()

# Load the image to overlay on the face
overlay = pygame.image.load("clown.png")

# Define a scaling factor (you can adjust this value as needed)
scaling_factor = 1.5

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Check if at least one face is detected
    if faces:
        # Use the first detected face for simplicity (you can modify this part if needed)
        face = faces[0]

        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # Extract the coordinates of the left and right eye
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Calculate the distance between the eyes
        eye_distance = int(((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2) ** 0.5)

        # Scale the overlay image based on the eye distance
        scaled_overlay = pygame.transform.scale(overlay, (
        int(eye_distance * scaling_factor), int(eye_distance * scaling_factor)))

        # Rotate the overlay image by 90 degrees
        rotated_overlay = pygame.transform.rotate(scaled_overlay, 90)

        # Get the position to overlay the image on the face
        overlay_position = (left_eye[0] - eye_distance // 4, left_eye[1] - eye_distance // 4)

        # Check if the overlay fits within the frame
        if (0 <= overlay_position[1] < frame.shape[0] and
                0 <= overlay_position[0] < frame.shape[1] and
                overlay_position[1] + rotated_overlay.get_height() <= frame.shape[0] and
                overlay_position[0] + rotated_overlay.get_width() <= frame.shape[1]):
            # Convert the Pygame surface to a NumPy array
            overlay_np = pygame.surfarray.array3d(rotated_overlay)

            # Blend the overlay image onto the original frame
            frame[overlay_position[1]:overlay_position[1] + rotated_overlay.get_height(),
            overlay_position[0]:overlay_position[0] + rotated_overlay.get_width()] = \
                frame[overlay_position[1]:overlay_position[1] + rotated_overlay.get_height(),
                overlay_position[0]:overlay_position[0] + rotated_overlay.get_width()] * \
                (1 - overlay_np / 255.0) + \
                overlay_np * (overlay_np / 255.0)

    # Display the frame
    cv2.imshow("Clown detector", frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
