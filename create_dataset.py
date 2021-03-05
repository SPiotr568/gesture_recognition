import cv2
import time
import beepy
import os
import imutils
from threading import Thread

camera = cv2.VideoCapture(0)

last = time.time()
number = 0
delay = 0.5
labels = []
quit_flag = False
preparation_time = 15

def get_labels():
    paths = [x[0] for x in os.walk('./images/')]
    for path in paths:
        if os.path.isdir(path) and path.split("/")[2] != '':
            labels.append(path.split("/")[2])
    print(f'Labels: {labels}')

def take_photo(label):
    name = "images/" + label + "/" +str(time.time()) + ".png"
    part_of_frame = frame[150:650, 100:600]
    #part_of_frame = cv2.cvtColor(part_of_frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(name,part_of_frame)
    beepy.beep(sound=1)

# Loading all labels
get_labels()

for label in labels:
    if quit_flag:
        break
    number = 0
    first_flag = True
    while number < 500:
        ret, frame = camera.read()
        t = time.time()
        diff = t - last
        cv2.putText(frame, "Put your hand here:", (100,140), cv2.FONT_HERSHEY_COMPLEX,  0.8, (0,0,255), 3)
        x, y = 96, 146
        w, h = 508, 508
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 3)
        text1 = "Label: '" + label + "'. Photo number: " + str(number+1)
        cv2.putText(frame, text1, (20,50), cv2.FONT_HERSHEY_COMPLEX,  1.4, (0,0,255), 3)
        next_photo = round(delay-diff, 2)

        if first_flag and diff < preparation_time:  
            next_photo = round(preparation_time-diff, 2)
        else:
            first_flag = False

        if next_photo < 0:
            next_photo = 0
        text2 = "Next: "  + str(next_photo) + "s"
        cv2.putText(frame, text2, (20,100), cv2.FONT_HERSHEY_COMPLEX,  1.4, (0,0,255), 3)

        if diff > delay and first_flag == False:
            thread = Thread(target = take_photo, args = (label,))
            thread.start()
            number += 1
            last = t
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) == ord('q'):
            quit_flag = True
            break
        elif cv2.waitKey(10) == ord('n'):
            break

camera.release()
cv2.destroyAllWindows()