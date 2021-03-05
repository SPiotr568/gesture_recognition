import numpy as np
import cv2
import os
from tensorflow import keras

text = ''
cap = cv2.VideoCapture(0)
model = keras.models.load_model('models/gesture_rec_model2.h5')
labels = []

def get_labels():
    paths = [x[0] for x in os.walk('./images/')]
    for path in paths:
        if os.path.isdir(path) and path.split("/")[2] != '':
            labels.append(path.split("/")[2])
    print(f'Labels: {labels}')

# Loading all labels
get_labels()

while True:
    text = ''
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    part_of_frame = gray[150:650, 100:600]
    img = cv2.resize(part_of_frame, (250, 250))
    array = np.array(img, dtype="uint8")
    ar = array.reshape(1,250,250,1)
    prediction = model.predict(ar)
    pred_max = 0
    for l in labels:
        pred_l = round(prediction[0][labels.index(l)],2)
        if pred_l > pred_max:
            pred_max = pred_l
            pred_text = "Prediction: " + l
        text += l + ': ' + str(pred_l) + ' '
    if pred_max < 0.8:
        pred_text = 'Not sure...'
    cv2.putText(frame, pred_text, (100,100), cv2.FONT_HERSHEY_COMPLEX,  1.2, (0,0,255), 2)
    cv2.putText(frame, "Put your hand here:", (100,140), cv2.FONT_HERSHEY_COMPLEX,  0.6, (0,0,255), 1)
    x, y = 96, 146
    w, h = 508, 508
    cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 3)
    cv2.putText(frame, text, (20,50), cv2.FONT_HERSHEY_COMPLEX,  1.4, (0,0,255), 2)
    cv2.imshow("Video",frame)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()