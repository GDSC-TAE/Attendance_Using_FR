

import numpy as np
import cv2
import pickle

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label={}
with open("labels.pickle",'rb') as f:
    orig_labels= pickle.load(f)
    label={v:k for k,v in orig_labels.items()} #Inverting Dictionary
    
    
capture=cv2.VideoCapture(0)
while(True):
    #capture video
    ret, frame= capture.read()
    #Convert color to gray
    grayclr= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= cascade_face.detectMultiScale(grayclr, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)                    #printing positional pixels
        gray_roi=grayclr[y:y+h , x:x+w]   # Co-ordinates == {(y_start,Y_end),(x_start,x_end)}
        gray_clr=frame[y:y+h , x:x+w]
        
        id,conf= recognizer.predict(gray_roi)
        if conf>= 45 and conf<=85:
            print(id)
            print(label[id])
                       
            
        rec_color=(0,0,255)               #Blue Green Red
        brush=3
        width=x+w
        height=y+h
        cv2.rectangle(frame, (x,y),(width,height),rec_color,brush)
        
    #show result
    cv2.imshow("video",frame)
    if cv2.waitKey(20)&0xFF== ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()    