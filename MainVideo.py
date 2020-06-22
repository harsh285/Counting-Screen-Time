import cv2
import numpy as np
import os 
from sklearn.externals import joblib
from keras.models import load_model
from sklearn.preprocessing import Normalizer



def gettingimage(img,model_facenet,in_encoder):
    img = cv2.resize(img,(160,160))
    # scale pixel values
    img = img.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    # transform face into one sample
    samples = np.expand_dims(img, axis=0)
    # make prediction to get embedding
    yhat = model_facenet.predict(samples)
    # normalize input vectors
    yhat = in_encoder.transform(yhat)
    return yhat;

# Load the  saved pre trained mode
model,out_encoder = joblib.load('saved_model.pkl')
faceCascade = cv2.CascadeClassifier('D:/Tools/Cascades/haarcascade_frontalface_default.xml')
model_facenet = load_model('facenet_keras.h5')
in_encoder = Normalizer(norm='l2')

# Initialize and start the video frame capture from webcam
dict1={'amitabh bachchan': int(0) ,'saira banu': int(0) ,'Sulakshana Pandit' : int(0) ,'Vinod Khanna': int(0) }
unknown=0
i=0

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture("Hera Pheri (1976).mp4")
fps = cam.get(cv2.CAP_PROP_FPS)
# Looping starts here
while True:
    i=i+1
    print(i)
    # Read the video frame
    ret, im =cam.read()
    # Convert the captured frame into RGB
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default

    # For each face in faces, we will start predicting using pre trained model
    for(x,y,w,h) in faces:
        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        # Recognition by SVM
        face_emb = gettingimage(gray[y:y+h,x:x+w],model_facenet,in_encoder)
        samples=face_emb
        
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        # Set rectangle around face and name of the person
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(predict_names), (x,y-40), font, 1, (255,255,255), 3)
        cv2.putText(im, str(class_probability), (x-40,y), font, 1, (255,255,255), 3)
        if(class_probability > 65):
            dict1[predict_names[0]]+=1
        else:
            unknown+=1

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video
cam.release()

# Close all windows
cv2.destroyAllWindows()
































