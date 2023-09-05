import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3-wider_16000.weights", "yolov3-face.cfg")

# Load COCO names
with open("face.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is for default camera

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width, channels = img.shape
    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Webcam Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
