import json
from datetime import datetime
import argparse

# Determining the current time in ISO 8601 format
iso_date = datetime.now().isoformat()


with open("carpark-spaces.json", "r") as temp_input:
    temp_data = json.load(temp_input)
    total_available_spaces_CP1 = temp_data["carpark"][0]["available_spaces"]
    total_available_spaces_CP2 = temp_data["carpark"][1]["available_spaces"]

parser = argparse.ArgumentParser('temp')
parser.add_argument('-r', '--reset', type=int, nargs='*', help='reset the thing to values')
args = parser.parse_args()

if args.reset is not None:
    total_available_spaces_CP1 = args.reset[0]
    total_available_spaces_CP2 = args.reset[1]


# this needs to be changed
# total_available_spaces_CP1 = 300
# total_available_spaces_CP2 = 300

with open('carpark1.json', 'r') as input_file_CP1:
    data_CP1 = json.load(input_file_CP1)

with open('carpark2.json', 'r') as input_file_CP2:
    data_CP2 = json.load(input_file_CP2)

available_spaces_CP1 = total_available_spaces_CP1 - data_CP1['in_count'] + data_CP1['out_count']

available_spaces_CP2 = total_available_spaces_CP2 - data_CP2['in_count'] + data_CP2['out_count']

# Data to be written to the JSON file 'EE399_CarCountingData.json'
data = {
    "carpark": [
                {
                    "id": 1,
                    "name": data_CP1['name'],
                    "available_spaces": available_spaces_CP1
                },
                {
                    "id": 2,
                    "name": data_CP2['name'],
                    "available_spaces": available_spaces_CP2
                }
             ],
    "date": iso_date
}

# Serializing json
json_object = json.dumps(data, indent=4)

# Writing to json file
with open("carpark-spaces.json", "w") as outfile:
    outfile.write(json_object)
