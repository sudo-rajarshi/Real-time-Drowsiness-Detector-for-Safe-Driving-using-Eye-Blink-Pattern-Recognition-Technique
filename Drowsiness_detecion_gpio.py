from __future__ import division
import curses
import RPi.GPIO as GPIO
import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM)

motor_L1 = 19 #35
motor_L0 = 26 #37
motor_R1 = 13 #33
motor_R0 = 21 #40

GPIO.setup(motor_L1, GPIO.OUT)
GPIO.setup(motor_L0, GPIO.OUT)
GPIO.setup(motor_R1, GPIO.OUT)
GPIO.setup(motor_R0, GPIO.OUT)

# Get the curses window, turn off echoing of keyboard to screen, turn on
# instant (no waiting) key response, and use special values for cursor keys
screen = curses.initscr()
curses.noecho()
curses.cbreak()
curses.halfdelay(3)
screen.keypad(True)

def forward():
    GPIO.output(motor_L1, GPIO.HIGH)
    GPIO.output(motor_L0, GPIO.LOW)
    GPIO.output(motor_R1, GPIO.HIGH)
    GPIO.output(motor_R0, GPIO.LOW)
    return forward

def backward():
    GPIO.output(motor_L1, GPIO.LOW)
    GPIO.output(motor_L0, GPIO.HIGH)
    GPIO.output(motor_R1, GPIO.LOW)
    GPIO.output(motor_R0, GPIO.HIGH)
    return backward

def right():
    GPIO.output(motor_L1, GPIO.LOW)
    GPIO.output(motor_L0, GPIO.LOW)
    GPIO.output(motor_R1, GPIO.HIGH)
    GPIO.output(motor_R0, GPIO.LOW)    
    return right

def left():
    GPIO.output(motor_L1, GPIO.HIGH)
    GPIO.output(motor_L0, GPIO.LOW)
    GPIO.output(motor_R1, GPIO.LOW)
    GPIO.output(motor_R0, GPIO.LOW)
    return left

def brake():
    GPIO.output(motor_L1, GPIO.LOW)
    GPIO.output(motor_L0, GPIO.LOW)
    GPIO.output(motor_R1, GPIO.LOW)
    GPIO.output(motor_R0, GPIO.LOW)
    return brake

def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("sound.ogg")
    pygame.mixer.music.play()


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation = interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation = interpolation)
        return resized


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36, 48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C) # ear = eye

    # return the eye aspect ratio
    return ear



#camera = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total = 0
alarm = False
camera = cv2.VideoCapture(0)
    
while True:
    ret,frame = camera.read()
    char = screen.getch()
    if char == ord('q'):
        break
    elif char == curses.KEY_UP:
        print("up")
        forward()     
    elif char == curses.KEY_DOWN:
        print("back")
        backward()
    elif char == curses.KEY_RIGHT:
        print("right")
        right()
    elif char == curses.KEY_LEFT:
        print("left")
        left()
    elif char == ord('b'):
        print("break")
        brake()
    

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
            if ear > .25:
                total = 0
                alarm = False
                cv2.putText(frame, "Eyes Open ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                total += 1
                if total > 5:
                    if not alarm:
                        brake()
                        alarm = True
                        d = threading.Thread(target=start_sound)
                        d.setDaemon(True)
                        d.start()
                        print("Stay Alert")
                        cv2.putText(frame, "drowsiness detected", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                cv2.putText(frame, "Eyes close".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break


curses.nocbreak()
screen.keypad(0)
curses.echo()
curses.endwin()
GPIO.cleanup()
