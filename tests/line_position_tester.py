import cv2
from lib.line_counter import Point, LineZone, LineZoneAnnotator
from deep_sort_realtime.deepsort_tracker import DeepSort

# supervision imports
from supervision.video import VideoInfo
from supervision.video import get_video_frames_generator
from supervision.video import VideoSink

import argparse


parser = argparse.ArgumentParser(description='Counting from a video')
parser.add_argument('filename', type=str, help='filename for the video(inlude the file extension, preferably .mp4)')
parser.add_argument('--line', type=int, nargs='*', default=[1920, 400, 0, 400], help='set the coords of the counting line')

# Parse the arguments
args = parser.parse_args()


INPUT_VIDEO = args.filename
video_info = VideoInfo.from_video_path(INPUT_VIDEO)

# Defining counting line
lineStart = Point(args.line[0], args.line[1])
lineEnd = Point(args.line[2], args.line[3])
lineCounter = LineZone(start=lineStart, end=lineEnd)
drawLine = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

cap = cv2.VideoCapture(INPUT_VIDEO)


while True:
    hasFrame, frame = cap.read()  # Obtaining a frame from the video

    if hasFrame == False:  # If there is no remaining frames then break out of for-loop
        break

    drawLine.annotate(frame=frame, line_counter=lineCounter)
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('custom window', frame)
    cv2.resizeWindow('custom window', 1280, 720)
    key = cv2.waitKey(1)
    if key & 0xff == 27:
        break
