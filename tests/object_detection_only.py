import torch
import cv2
import numpy as np

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

INPUT_VIDEO = args.filename
OUTPUT_VIDEO = 'OUTPUT-' + DEVICE + '-' +  datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-") + INPUT_VIDEO
# OUTPUT_VIDEO = 'OUTPUT_'+INPUT_VIDEO

video_info = VideoInfo.from_video_path(INPUT_VIDEO)
generator = get_video_frames_generator(INPUT_VIDEO)


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
            class_name = classes[class_id]
        if class_name in allowed_objects:
            (x1, y1, x2, y2) = box
            detectionsList.append(([x1, y1, x2 - x1, y2 - y1], score, classes[class_id]))

            # printing the bounding box, class name and score
            text = str(class_name + ": " + str(score))
            cv2.putText(frame, text, (int(x1), int(y1-4)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)


        frame_time = time.time() - start_time
        frame_fps = 1/frame_time
        print("%0.2f fps" % (frame_fps), end=' ')

        # calculate power usage
        power = subprocess.run(["cat", "/sys/class/power_supply/BAT0/power_now"], capture_output=True)
        power = (int(power.stdout)/1e6)
        print(str(power) + "W", end=' ')

        print("%0.2f%%" % (100*i/video_info.total_frames), end=' ')
        i = i + 1
        
        # Adding a title to the frame
        cv2.putText(frame, "Object Detection Only (YOLOv5)", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 2)
        
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
