# Face Recognition on still images

import cv2
import numpy as np
a = 0
ad = []
# Create a file with paths on your PC
file = open('paths', 'r')
for i in file.readlines():
    ad.append(i)

for j in range(len(ad)):
    if j == len(ad)-1:
        path = ad[j]
    else:
        path = ad[j][:-1]
    print(path)
    # Download the .xml file and change the path
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = cv2.imread(path, 1)

    try:
        image = cv2.resize(image, (600, 550))
    except cv2.error:
        pass

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    search_face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5)

    if type(search_face) != type(()):
        for x in search_face:
            a+=1
            image = cv2.rectangle(image, (x[0], x[1]), (x[0] + x[2], x[1] + x[3]), (205, 25, 17), 3)
            name = str(a)+'.png'
            cropped_img = image[x[1]:x[1] + x[3], x[0]:x[0] + x[2]]
            cv2.imwrite(name, cropped_img)

        cv2.imshow('final_image', image)

        print('Face has been detected')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('No Face Detected')
