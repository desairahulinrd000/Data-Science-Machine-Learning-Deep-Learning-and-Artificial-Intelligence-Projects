import numpy as np
from PIL import Image
import os
import cv2
def train_classifier(data_dir):
    path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces=[]
    ids=[]
    for images in path:
        img=Image.open(images,).convert('L')
        imageNp=np.array(img,'uint8')
        id=int(os.path.split(images)[1].split(".")[1])
        
        faces.append(imageNp)
        
        ids.append(id)
        
    ids=np.array(ids)
 
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.save("Classifier2.yml")
    print("Model for Your Face Has been preapered now time to Recognize yourself by a machine")
train_classifier("Training Images")
