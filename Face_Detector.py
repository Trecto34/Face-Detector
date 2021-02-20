import cv2
from random import randrange

# Load the pre-trained data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Image that the face will be detected
img = cv2.imread("./images/mask.jpeg")

# Convert the image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw retangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# print(face_coordinates)

# Display the image
cv2.imshow("Clever Programmer Face Detector", img)
cv2.waitKey()

print("Code Completed")
