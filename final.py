#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install retina-face
# !pip install deepface
import os
import shutil    #to delete a whole directory with files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from retinaface import RetinaFace   #to extract indiviual faces from image
import uuid      #to generate random image names unique
import time

#getting paths
current_dir = os.getcwd()
input_path = os.path.join(current_dir, 'input_img')
verification_path = os.path.join(current_dir, 'verification_img')
print(verification_path)

try:
    print("Creating Folder For Verified Image.")
    os.mkdir(verification_path)
except FileExistsError:
    pass


# In[ ]:


def take_picture():
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        ret, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:

            # Save the image when face is detected
            cv2.imwrite('grp.jpg', img)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


        # Display the output
        cv2.imshow('Video Face Detect', img)

        # Exit on ESC key press
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()

take_picture()


# In[4]:


try:
    os.mkdir(input_path)
except FileExistsError:
    print("Removing the existing folder.")    
    shutil.rmtree(input_path)
    os.mkdir(input_path)
print('Input Image Folder Created.')
#extracting each face from the image
faces = RetinaFace.extract_faces(img_path = "grp.jpg", align = True)
if len(faces) == 0:
    print("No face detected in the image.\nRe-taking The Image.")
    take_picture()
    
#making separate jpg file for each face
for face in faces:
    img_path = os.path.join(input_path, '{}.jpg'.format(uuid.uuid1()))
    # print(img_path)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, face)


# In[6]:


backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace','DeepID','ArcFace', 'Dlib', 'SFace'] #face verification

metrics = ['cosine', 'euclidean', 'euclidean_l2']

all_input_imgs = os.listdir(input_path)

print(all_input_imgs)

fig = plt.figure(figsize=(10, 5))

val = 0  #initialized to display the image on different subareas

#Displays the viable faces that can be used for face recognition

for i in all_input_imgs:
    # img_path = input_path + f"\\{i}"
    val += 1
    img_path = os.path.join(input_path,f'{i}')
    print(img_path)
    try:
        face_detection = DeepFace.detectFace(img_path, detector_backend = 'mtcnn')
        fig.add_subplot(1, len(all_input_imgs),val)
        plt.imshow(face_detection)
        plt.axis('off')
    except:
        os.remove(img_path)
    


# In[7]:


# file_name = []
status = False

#Performing face recognition using FaceNet
for i in os.listdir(input_path):
    img_path = os.path.join(input_path,f'{i}')
    print(img_path)
    output = DeepFace.find(img_path, verification_path, model_name=models[1], distance_metric= metrics[0], enforce_detection= True, detector_backend = backends[3])
    print(output)
    is_there = output[0]['identity']
    is_there = list(is_there)
    print(is_there)
    if len(is_there) != 0:
        # file_name.append(is_there)
        status = True
        file_name = is_there[0]    
        break
 


# In[9]:


#Simply plotting the input face and the face from database if present
#Performing Slicing to find out the identity of the face
try:
    print(file_name)
    # img_arr = cv2.imread(file_name)
    # plt.imshow(img_arr)
    grab_element = is_there[0]
    # print(grab_element)
    words = grab_element.split('/')
    name = words[-1]
    # print(name)
    info = name[:-4]
    print(info)

    fig,axs = plt.subplots(1,2, figsize = (15,5))

    axs[0].imshow(plt.imread(img_path))
    axs[1].imshow(plt.imread(file_name))
    fig.suptitle(f"Owner Verification: {status}\n Owner Identity: {info}")
    plt.show()
except:
    # print('An Intruder Is Trying To Enter Your House.')
    fig,axs = plt.subplots(1,2, figsize = (15,5))

    axs[0].imshow(plt.imread('grp.jpg'))
    fig.suptitle(f"Owner Verification: {status}\n Intruder Detected!")
    plt.show()
    



# In[ ]:


#Using smtp to send automail
import smtplib
import imghdr

from email.message import EmailMessage

email_id = 'rdplaysgg@gmail.com'
email_pass = 'rvpxjqrqpqcoxkiu'

msg = EmailMessage()

def mail_now(identity, path):
    msg['Subject'] = "Security Update!!"
    msg['From'] = email_id
    msg['To'] = 'rjtdulal@gmail.com'
    msg.set_content(f"Here is the picture of the individual trying to enter your house.\n\n Status: {identity} MEMBER ")

    with open(path) as m:
        file_data = open(path, 'rb').read()
        file_type = imghdr.what(m.name)
        # file_name = m.name
        file_name = "Image of Individual"
        print(file_name)

    msg.add_attachment(file_data, maintype = 'image',subtype = file_type, filename = file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        smtp.login(email_id, email_pass)
        smtp.send_message(msg)


if status == True:
    identity = "Verified".upper()
    path = img_path
    # path = 'grp.jpg'

else:
    identity = "Unverified".upper()
    path = 'grp.jpg'

try:
    mail_now(identity, path)
except:
    print("No Internet Access. Connect To Internet First")
      


# In[ ]:




