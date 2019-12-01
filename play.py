from keras.models import load_model
import cv2
import numpy as np
from time import sleep
model = load_model("gesturecheck.h5")
z=0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user1 to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for user2 to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
    edges = cv2.Canny(frame,100,200)
    roi1=edges[100:500, 100:500]
    roi2=edges[100:500, 800:1200]
    img1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (227, 227))
    pred1 = model.predict(np.array([img1]))
    move_code1 = np.argmax(pred1[0])
    img2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (227, 227))
    pred2 = model.predict(np.array([img2]))
    move_code2 = np.argmax(pred2[0])
    x=int(move_code1)
    y=int(move_code2)
    printerr=''
    if x==y:
    	z=0
    	printerr='game over'
    else:
    	z+=x
    	printerr='score = '+str(z)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, printerr,
               (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Hand Cricket", frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
