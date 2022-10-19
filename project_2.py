#TRAINING FACES
import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
img_dir= os.path.join(BASE_DIR, "media")
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#recognizer= cv2.face.LBPHFaceRecognizer_create()

y_labels= []
x_train= []
current_id=0
label_id={}

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            #print(label,path)
            
            if not label in label_id:
                label_id[label]=current_id
                current_id+=1
                
            id= label_id[label]
            #print(label_id)
                
            #we want labels as numbers and images as numpy arrays.
            pil_image = Image.open(path).convert("L")# gryscaling image
            img_array= np.array(pil_image, "uint8")#converting images to numpy arrays
            #print(img_array)
            
            faces = cascade_face.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = img_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id)

print(x_train)
print(y_labels)

#with open("labels.pickle",'wb') as f:
#    pickle.dump(label_id, f)
    

#recognizer.train(x_train, np.array(y_labels))
#recognizer.save("trainer.yml")