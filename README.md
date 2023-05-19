# Information
This repo contains code that was used to demonstrate the viability of an intelligent car parking solution for EE399, a community based project module in the Department of Electronic Engineering, Maynooth University.

Click below for an example of the system in action.

[![Example Video of System](https://img.youtube.com/vi/apLurdmOGik/0.jpg)](https://www.youtube.com/watch?v=apLurdmOGik)

There may (and likely are!) issues with running the code as is. There is very little documentation as of now.

## Installing dependencies
Ensure you have Python>=3.8 installed.

Perform the following command in a command line or terminal:
```bash
pip3 install -r requirements.txt
```
This should install the necessary Python libraries to run the code.

## Running the code
You will need to supply a video to test the system on. 

You can then run the system with the following command:
```bash
python3 src/car_counter.py path/to/video/file.mp4 --video
```
This should (hopefully!) run the car counting on the video, and produce an output video similar to the one above in the same directory as the command was run from. There should be information about the progress printed on the terminal, and a `.csv` file should be generated with information about frame time and instantaneous fps.