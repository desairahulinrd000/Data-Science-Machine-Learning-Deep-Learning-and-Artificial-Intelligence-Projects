import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Classifier2.yml")
cap = cv2.VideoCapture(0)
ret=True
while ret:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        if conf>=40:
            font=cv2.FONT_HERSHEY_COMPLEX
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,"Your Name",(x,y-4),font,1,color,stroke,cv2.LINE_AA)
        color=(255,0,0)
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1)==27:
        break       
cap.release()
cv2.destroyAllWindows()
