import cv2
from time import sleep

def generate_dataset(img,id1,img_id):
    global count
    count-=1
    cv2.imwrite("Training Images/Myself_"+str(id1)+"."+str(img_id)+".jpg",img)

def draw_boundary(img,classifier, scaleFactor, minNeighbors, color, text):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    coords=[]
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img, text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x,y,w,h]
        
        
    return coords


def detect(img,faceCascade,img_id):
    color={"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    
    coords=draw_boundary(img, faceCascade,1.5,5,color["blue"],"Face")
    
    
    
    if len(coords)==4:
        roi_img=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        user_id=1
        
        generate_dataset(roi_img,user_id,img_id)
    
    return img


face_cascade=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")

video_capture=cv2.VideoCapture(0)
img_id=0
ret=True
print("Face yourself in front camera till camera window closes")
sleep(2)
count=200
while ret and count>0:
    if video_capture.isOpened():
        ret,img=video_capture.read()
        img=detect(img,face_cascade,img_id)
        cv2.imshow("Face Detection",img)
        img_id+=1
    else:
        ret=False
    if cv2.waitKey(1)==27:
        break
if count<=0:
    print("Training Images Created Successfully, now build the training model")   
else:
    print("Failed to build the training images try again")     
video_capture.release()
cv2.destroyAllWindows()
