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
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"


    cv2.imshow("object Detect", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
