# from time import sleep
import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# supervision imports
# from supervision.draw.color import ColorPalette
# from supervision.geometry.core import Point
from supervision.video import VideoInfo
from supervision.video import get_video_frames_generator
from supervision.video import VideoSink
# from supervision.detection.core import Detections, BoxAnnotator
# from supervision.detection.line_counter import LineZone, LineZoneAnnotator

import argparse
import subprocess
import datetime
import time
import csv
import json

# change to laptop or nano depending on test device
DEVICE = 'laptop'

# change depending on carpark (may change this to an argument)
CARPARK_ID = 2
CARPARK_NAME = 'Car Park 2'

# Create the parser
parser = argparse.ArgumentParser(description='Counting from a video')

# Add the argument
parser.add_argument('filename', type=str, help='filename for the video(inlude the file extension, preferably .mp4)')

# Parse the arguments
args = parser.parse_args()

# Access the argument
# print('mystring:', args.filename)

# ----------------------------------------------------------

# load model and set up allowed classes
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
CLASS_NAMES = model.model.names
ALLOWED_OBJECTS = ["car", "motorcycle", "bus", "truck"]

# initialise tracker
object_tracker = DeepSort(
    max_age=12,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

INPUT_VIDEO = args.filename
OUTPUT_VIDEO = 'OUTPUT-' + DEVICE + '-' +  datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-") + INPUT_VIDEO
# OUTPUT_VIDEO = 'OUTPUT_'+INPUT_VIDEO

video_info = VideoInfo.from_video_path(INPUT_VIDEO)
generator = get_video_frames_generator(INPUT_VIDEO)

# filename = args.filename + '.mp4'
# Loading Video File
# cap = cv2.VideoCapture("output.mp4")
cap = cv2.VideoCapture(INPUT_VIDEO)

i = 1
frame_list = []
time_list = []
fps_list = []
power_list = []

with VideoSink(OUTPUT_VIDEO, video_info=video_info) as sink:
    while True:
        start_time = time.time()

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
            cv2.putText(frame, "ID: " + str(track_id), (int(box[0]), int(box[1] - 6)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

        
        # Adding a title to the frame
        cv2.putText(frame, "Object Detection and Tracking (YOLOv5 + DeepSORT)", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 2)

        frame_time = time.time() - start_time
        frame_fps = 1/frame_time
        print("%0.2f fps" % (frame_fps), end=' ')

        # calculate power usage
        power = subprocess.run(["cat", "/sys/class/power_supply/BAT0/power_now"], capture_output=True)
        power = (int(power.stdout)/1e6)
        print(str(power) + "W", end=' ')

        print("%0.2f%%" % (100*i/video_info.total_frames), end=' ')
        i = i + 1
        sink.write_frame(frame=frame)
        
        frame_list.append(i-1)
        time_list.append(frame_time)
        fps_list.append(frame_fps)
        power_list.append(power)


cap.release()
cv2.destroyAllWindows()

# csv output for testing
csv_filename = DEVICE + '-' +  datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-") + INPUT_VIDEO + '.csv'
with open(csv_filename, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header row to the CSV file
    writer.writerow(['Frame', 'time (s)', 'fps', 'power (W)'])

    # write data to file
    writer.writerows(zip(frame_list, time_list, fps_list, power_list))
