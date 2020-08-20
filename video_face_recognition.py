# Python OpenCV Live face detection

import time
import cv2
import numpy

start_time = time.time()

video_frames = cv2.VideoCapture(0)

# setting initial frame
a = 1

while True:
    # Frame increment
    a += 1
    # Boolean, Frame
    para, frame = video_frames.read()
    # Printing numpy, matrix
    print(frame)
    # Initializing Face Cascade
    # Download the .xml file and change the path according to your PC
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Frame conversion RGB to GRAY scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection co-ordinates
    face_detect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

    # Plotting face using rectangle
    for x in face_detect:
        frame = cv2.rectangle(frame, (x[0] - 75, x[1] - 20), (x[0] + x[2] - 10, x[1] + x[3] + 20), (205, 25, 17), 3)
        # Cropping the face frame and saving it in the CWD
        cropped_frame = frame[x[1] - 20:x[1]+x[3] + 20, x[0] - 75:x[0]+x[2] - 10]

        name = 'croppedimg'+str(a)+'.png'

        cv2.imwrite(name, cropped_frame)

    cv2.imshow('Frame', frame)

    # 1 ms delay
    key = cv2.waitKey(1)

    # Press e to exit
    if key == ord('e'):
        break

# Frame analysis / Performance Analysis
print('Total Frames =', a)
end_time = time.time()
print('Total_time = ', end_time-start_time)
print('Frames per second', a/(end_time-start_time))


