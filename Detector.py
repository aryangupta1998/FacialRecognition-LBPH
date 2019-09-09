import cv2
import numpy as numpy

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
print(rec.read('trainingData.yml'))
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
identified=[]
while True:
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if id==1:
            cv2.putText(img,"Aryan",(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
            identified.append("16BCE2155")
        elif id ==2:
            cv2.putText(img,"Shubham",(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
            identified.append("16BEI0046")
        elif id ==3:
            cv2.putText(img,"NAVIN",(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
            identified.append("16BCE0130")
        else:
            cv2.putText(img,str(id),(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
        pass
    cv2.imshow("Faces",img)
    if(cv2.waitKey(1) == ord('q')):
        break
    pass

#identified.append("16BCE0130")
#identified.append("16BCE0102")
#identified.append("16BCE0682")
#identified.append("16BCI0203")
present=list(set(identified))
print("ATTENDANCE SYSTEM\n")
print("IDs of students who are present:")
for person in present:
    print(person)

cam.release()
cv2.destroyAllWindows()
