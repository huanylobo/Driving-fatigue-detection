from __future__ import division
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
import pygame

import cv2
import dlib
import numpy as np
import time


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36, 48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total = 0
alarm = True

allFrame=0
yawn_time = time.time()
mouth_opentime=yawn_time
mouth_closetime=yawn_time
yawn_decide_time=0.00
pCounter=0
while True:
    #creat a new frame counter to count every 10 frames(for PERCLOS)
    if allFrame % 10 ==0:
        pCounter=0
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)
    prev_yawn_status = yawn_status
    if lip_distance > 20:
        mouth_opentime=time.time()
        yawn_status = True
        cv2.putText(frame, "you are Yawning", (50, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        yawn_status = False
        mouth_closetime=time.time()

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1
    output_text = " Yawn Count: " + str(yawns)
    cv2.putText(frame, output_text, (50, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
    # cv2.imshow('Live Landmarks', image_landmarks)

    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear > .20:
                print("EAR is :")
                print(round(ear, 2))
                # total=0
                alarm = False
                cv2.putText(frame, "Eyes Open ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 204, 0), 2)
            else:
                total += 1
                pCounter+=1
                if total > 20:
                    if not alarm:
                        alarm = True
                        cv2.putText(frame, "drowsiness detect", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                cv2.putText(frame, "Eyes close".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
    print("total blinks number is"+str(total))
    allFrame=allFrame+1
    #when PERCLOS is higher than 0.4,the object is fatigue.
    PERCLOS=pCounter/10
    yawn_decide_time=mouth_opentime-mouth_closetime
    if (PERCLOS>=0.4) | (yawn_decide_time>=3.00):
        cv2.putText(frame,"tired now",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2)
    cv2.putText(frame, "blink number:"+str(total), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    format_yawn_time = "{:.2f}".format(yawn_decide_time)
    print("yawn time is "+format_yawn_time)
    # cv2.imshow("image", frame)
    cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
