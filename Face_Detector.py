import cv2
from random import randrange

# Load the pre-trained data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detectInImage(imageSource):
    # Image that the face will be detected
    img = cv2.imread(imageSource)

    # Convert the image to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw retangles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Clever Programmer Face Detector", img)
    cv2.waitKey()
    print("Detect in image Completed")


def detectInVideo(sourcePath=0):  # By default this will capture the webcam
    # Capture the default Webcam
    webcam = cv2.VideoCapture(sourcePath)
    key = cv2.waitKey(1)

    # Interact forever over frames
    while True:
        # Read the current frames
        succesful_frame_read, frame = webcam.read()

        # Convert the frame to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

        # draw retangles around the face
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Clever Programmer Face Detector", frame)
        key = cv2.waitKey(1)

        # If press the "Q" key the programm will close
        if key == 81 or key == 113:
            break

    webcam.release()
    print("Detect in webcam finished")


if __name__ == "__main__":
    detectInImage("./images/GroupWithMask.jpeg")
    detectInImage("./images/Person.png")
    detectInImage("./images/Person2.png")
    detectInVideo("./images/video.mp4")  # Leave blank or "0" to use webcam,
