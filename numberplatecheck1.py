import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Coco class names  load
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer
layer_names = net.getUnconnectedOutLayersNames()

# video stream open (0 for default camera)
cap = cv2.VideoCapture("traffic.mp4")
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # preprocessing yolov3 frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    detections = net.forward(layer_names)
    # post process the detections
    for detection in detections: