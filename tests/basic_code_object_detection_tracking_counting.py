import torch
import cv2
import numpy as np
from lib.line_counter import Point, LineZone, LineZoneAnnotator
from deep_sort_realtime.deepsort_tracker import DeepSort

# load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# getting a list of the objects that YOLO can detect
CLASS_NAMES = model.model.names

# creating an array of objects of interest used to filter object detections
ALLOWED_OBJECTS = ["car", "motorcycle", "bus", "truck"]

# Loading Video File - Insert path location to Video File
cap = cv2.VideoCapture("TestVideo3_Recorded.mp4")

# DeepSort Tracker
object_tracker = DeepSort(max_age=15)

# Defining counting line
lineStart = Point(1690, 450)
lineEnd = Point(10, 450)
# Creating a line counting object
lineCounter = LineZone(start=lineStart, end=lineEnd)
drawLine = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

# Loading Video File - Insert path location to Video File
cap = cv2.VideoCapture("TestVideo3_Recorded.mp4")

while True:
    hasFrame, frame = cap.read()  # Obtaining a frame from the video

    if hasFrame == False:  # If there is no remaining frames then break out of for-loop
        break

    # Obtaining results of the model on the frame
    results = model(frame)

    # Converting model results to detections
    detections = results.pandas().xyxy[0]
    boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    confidence = detections['confidence'].values.tolist()
    class_ids = detections['class'].values.tolist()

    # Filtering detections
    detectionsList = []  # list of allowed detected objects in the form ([x,y,w,h], score, class)
    for (class_id, score, box) in zip(class_ids, confidence, boxes):
        class_name = CLASS_NAMES[class_id]
        if class_name in ALLOWED_OBJECTS:
            (x1, y1, x2, y2) = box
            detectionsList.append(([x1, y1, x2 - x1, y2 - y1], score, CLASS_NAMES[class_id]))

    # Updating the DeepSort Object tracker
    tracks = object_tracker.update_tracks(detectionsList, frame=frame)

    countList = []  # list of tracked objects in the form ([x1,y1,x2,y2], track_id)

    # for-loop iterates through the tracked objects
    for track in tracks:
        if track.is_confirmed() == False:
            continue
        track_id = track.track_id
        box = track.to_ltrb()
        (x1, y1, x2, y2) = box
        countList.append(([x1, y1, x2, y2], int(track_id)))

        # printing the bounding box and track ID on the tracked object
        text = str("ID: " + str(track_id))
        (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1 - 25)), (int(x1 + w_text), int(y1)), (200, 0, 0), -1)
        cv2.putText(frame, text, (int(box[0]), int(box[1] - 6)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    # Updating the line counter
    lineCounter.trigger(detections=countList)

    # Displaying the frame
    drawLine.annotate(frame=frame, line_counter=lineCounter)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key & 0xff == 27:    # press the 'Esc' key to stop the video
        break

cap.release()
cv2.destroyAllWindows()
