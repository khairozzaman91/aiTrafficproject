import cv2
import numpy as np
import time

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Coco class names  load
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer
layer_names = net.getUnconnectedOutLayersNames()

# video stream open (0 for default camera)
cap = cv2.VideoCapture("traffic.mp4")

previous_frame = None
previous_time = None
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if previous_frame is None:
        previous_frame = frame
        previous_time = time.time()
        continue

    # preprocessing yolov3 frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    detections = net.forward(layer_names)
    # post-process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'car':

                # car speed calculate
                current_time = time.time()
                time_difference = current_time - previous_time
                if time_difference > 0:
                    road_width_pixels = frame.shape[1]
                    pixel_distance = obj[2] * road_width_pixels
                    meters_per_pixel = 1.0

                    # convert to mph
                    speed_mph = (pixel_distance / time_difference) * meters_per_pixel * 2.23694
                    print(f"Car speed : {speed_mph : .2f} mph")

                previous_time = current_time
    # Display the frame with object detection
    cv2.imshow("Car speed : ", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
