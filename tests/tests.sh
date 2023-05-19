#!/bin/bash

# NOTE: this is a bit hard coded at the moment so it will not work unless you
# 		have a folder organised in the same way as the files below

folder="./test-videos"
device="laptop"
output_folder="./tests-csv"

echo "making folder $output_folder ......"
mkdir -p $output_folder

for file in "$folder/"*.mp4; do 
		if [[ -f "$file" ]]; then
				# echo "Processing: $file"
				command="python3 combined.py $file --device $device --out_folder $output_folder"
				echo "$command"
				# eval "$command" 
				echo "........................................"
				notify-send "Video Completed" "$file has been processed"
		fi
done

file="$folder/online/TestVideo1_Online_reencode.mp4"
command="python3 combined.py $file --device $device --out_folder $output_folder --line 1920 650 0 650"
echo "$command"
# eval "$command" 
echo "........................................"
notify-send "Video Completed" "$file has been processed"

file="$folder/online/TestVideo2_Online_reencode.mp4"
command="python3 combined.py $file --device $device --out_folder $output_folder --line 1920 650 0 650"
echo "$command"
# eval "$command" 
echo "........................................"
notify-send "Video Completed" "$file has been processed"

file="$folder/online/TestVideo3_Online_reencode.mp4"
command="python3 combined.py $file --device $device --out_folder $output_folder --line 1920 600 0 600"
echo "$command"
# eval "$command" 
echo "........................................"
notify-send "Video Completed" "$file has been processed"
notify-send "Finished" "All videos have been processed"
