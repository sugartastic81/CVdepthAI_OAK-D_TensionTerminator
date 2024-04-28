# Tension Terminator OfficePro T-800 | Ergophysion
# code by MCI DiBSE 2021 Group 1.2 "Tschäin"

# main programm with EasyGUI to run on a device supporting the DepthAI SDK

# run DepthAI SKD
# https://docs.luxonis.com/projects/sdk/en/latest/
# https://docs.luxonis.com/projects/sdk/en/latest/components/nn_component/#nncomponent

# Import necessary libraries
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from depthai import NNData
from datetime import datetime
from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections, DetectionPacket
from depthai_sdk.visualize.visualizer import Visualizer
from depthai_sdk.visualize.configs import BboxStyle, TextPosition


# Global variables to store exercise data and various states
exercise_data_list = []
headCOUNTER = 0
headlastTIME = 0
headlastX = 0
headlastY = 0
headlastZ = 0
head_savedZ = 0
headZ_movement_changes = 0
headZ_movement = False
exerciseSTART = 0
exerciseEND = 0
exerciseDURATION = 0
exerciseTYPE = "none"
exerciseAREAnow = "none"
exerciseAREAprev = "none"
exerciseAREAchangeTIME = 0
exerciseAREAchangeFLAG = False
exerciseRUNNING = False
breakFLAG = False

# Generate a timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d%H%M")
file_name = f"TTdata_{timestamp}.JSON"
folder_name = "TTdata"

'''
ATTENTION! In the YOLO8 model the classes are not named (bug) so we need to name them [] in the JSON file
WRONG:
     "mappings": {
         "labels": [
             "Class_0",
             "Class_1",
             "Class_2"
         ]
    },
CORRECT:
"mappings": {
    "labels": [
        "Duoballs",
        "Head",
        "Triggerpointlever"
    ]
},
'''

# - slow is the default mode
modeltext = "Slow mode with Yolo5m MEDIUM 320x320 selected."
modelJSON = "ModelZoo\TTmodel1_yolo5m_img320\TTmodel320v2m.json"

# Check if any command-line arguments were passed
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == "-fast":
            modeltext = "Fast mode with Yolo8s SMALL 320x320 selected. (lower object detection rate)"
            modelJSON = "ModelZoo\TTmodel2_yolov8s_img320\TTmodel2_yolov8s_img320_20231129_212044.json"

print("\n"+modeltext+"\n")

# Function to load labels from the JSON file
def load_labels(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        labels = data['mappings']['labels']
        return labels
NNlabels = load_labels(modelJSON)

# Function to append new data to the JSON file
def append_to_json_file(new_data):
    global file_name, folder_name
    full_file_path = os.path.join(folder_name, file_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Read existing data
    try:
        with open(full_file_path, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Append new data and write to file
    data.append(new_data)
    with open(full_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to decode neural network data
def decode(nn_data: NNData):
    print("Decoding the data ...")
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)

    for result in results[0][0]:
        if result[2] > 0.5:
            dets.add(result[1], result[2], result[3:])
            print(f"Detections: Class {result[1]}, Confidence {result[2]}, Coordinates {result[3:]}")

    return dets

# Function to clear the terminal screen for the EasyGUI
def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

# Callback function to process detection packets
def cb(packet: DetectionPacket):
    global headCOUNTER, headlastTIME, exerciseSTART, exerciseEND, exerciseDURATION, exerciseRUNNING, \
        exercise_data_list, exerciseTYPE, headlastX, headlastY, headlastZ, exerciseAREAnow, exerciseAREAprev, \
        exerciseAREAchangeTIME, exerciseAREAchangeFLAG, breakFLAG, headZ_movement, headZ_movement_changes, head_savedZ

    current_time = datetime.now()

    clear_screen()
    print("*********************************************")
    print("* TensionTerminator OfficePro | Ergophysion *")
    print("*********************************************\n")

    # Reset counter if more than 4 second since last head detection
    if headlastTIME and (current_time - headlastTIME).total_seconds() > 3.9:
        breakFLAG = True
    else:
        if exerciseRUNNING == False:
            exerciseSTART = current_time
            exerciseRUNNING = True
        exerciseEND = current_time
        exerciseDURATION = (exerciseEND - exerciseSTART).total_seconds()

    if exerciseRUNNING == False:
        exerciseDURATION = 0

    # Create a Visualizer instance
    #visualizer = Visualizer()
    # Prepare visualizer objects
    #packet.prepare_visualizer_objects(visualizer)
    # Get the frame with drawn objects
    #frame_with_objects = visualizer.drawn(packet.frame)

    spatial_detections = packet.img_detections

    # Print various exercise and detection related information
    print(f"Excercise Type:\t\t{exerciseTYPE}\tExcercise Running:\t{exerciseRUNNING}")
    print(f"Excercise Area Now:\t{exerciseAREAnow}\tExcercise Area Prev:\t{exerciseAREAprev}")
    print(f"Exercise Area Change:\t{exerciseAREAchangeTIME}\tExercise Change Flag:\t{exerciseAREAchangeFLAG}")
    print("Break Flag:\t\t", breakFLAG)

    print("\nExcercise Start:\t", exerciseSTART)
    print("Excercise End:\t\t", exerciseEND)
    print("Excercise Duration:\t", exerciseDURATION)
    print("Head Counter:\t\t", headCOUNTER)
    print("Time Last Detection:\t", headlastTIME)

    print(f"\nMovement Up&Down:\t{headZ_movement}\tHead Saved Z:\t{head_savedZ}")
    print(f"Movement Changes:\t{headZ_movement_changes}")

    print("\n<===================>\n")

    if hasattr(spatial_detections, 'detections') and breakFLAG != True:
        for detection in spatial_detections.detections:
            label_name = NNlabels[detection.label] if detection.label < len(NNlabels) else "Unknown"

            if(label_name == "Head"):
                headCOUNTER += 1
                headlastTIME = current_time

            confidence_percent = int(detection.confidence * 100)
            if headCOUNTER > 10:
                print(f"Object Detection:\tLabel: {label_name}\tConfidence: {confidence_percent}%\t\tTime: {current_time}")
                print(f"Bounding Box:\t\tx_min: {detection.xmin:.5f}\ty_min: {detection.ymin:.5f}\tx_max: {detection.xmax:.5f}\ty_max: {detection.ymax:.5f}")

                # if spatial data is available
                if hasattr(detection, 'spatialCoordinates') and label_name == "Head":
                    headlastY = detection.spatialCoordinates.y
                    headlastX = detection.spatialCoordinates.x
                    headlastZ = detection.spatialCoordinates.z

                    if( head_savedZ == 0 ):
                        head_savedZ = headlastZ

                    if abs(head_savedZ - headlastZ) > 80:
                        headZ_movement_changes += 1
                        head_savedZ = headlastZ
                        if headZ_movement_changes >= 4:
                            headZ_movement = True

                    if exerciseAREAnow != exerciseAREAprev:
                        exerciseAREAchangeFLAG = True
                    else:
                        exerciseAREAchangeFLAG = False
                        exerciseAREAchangeTIME = current_time
                    if( headlastY > 30 and headlastX < 70 ):
                        exerciseAREAnow = "Triggerpointlever"
                    elif( headlastY <= 30 and headlastX > -110 and headlastX < 110):
                        exerciseAREAnow = "Duoballs"
                    else:
                        exerciseAREAnow = "none"
                    # just makes a area change after 4 seconds
                    if (current_time - exerciseAREAchangeTIME).total_seconds() > 4 and exerciseAREAchangeFLAG == True:
                        if exerciseAREAprev != "none" and exerciseTYPE != exerciseAREAnow:
                            breakFLAG = True
                        else:
                            exerciseTYPE = exerciseAREAnow
                            headZ_movement_changes = 0
                        exerciseAREAprev = exerciseAREAnow
                    print(
                        f"Spatial Coordinates:\tx: {detection.spatialCoordinates.x:.5f}\ty: {detection.spatialCoordinates.y:.5f}\tz: {detection.spatialCoordinates.z:.5f}")


    # saves the exercise data to the list and resets the variables
    if (breakFLAG == True):
        if exerciseDURATION > 5: # Only saves it, if exercise was longer than 5 seconds
            new_entry = {
                "exercise": exerciseTYPE,
                "start": exerciseSTART.strftime("%Y-%m-%d %H:%M:%S"),
                "end": exerciseEND.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": exerciseDURATION,
                "movementsZ": round(headZ_movement_changes/2)
            }
            exercise_data_list.append(new_entry)
            # Write data to the file
            append_to_json_file(new_entry)

        headCOUNTER = 0
        headlastTIME = 0
        headlastX = 0
        headlastY = 0
        headlastZ = 0
        exerciseSTART = 0
        exerciseEND = 0
        exerciseDURATION = 0
        headZ_movement = False
        headZ_movement_changes = 0
        exerciseAREAprev = exerciseAREAnow
        exerciseTYPE = exerciseAREAprev
        exerciseAREAchangeTIME = 0
        exerciseAREAchangeFLAG = False
        exerciseRUNNING = False
        breakFLAG = False

    if (exerciseAREAnow == "none" and exerciseTYPE == "none" and exerciseDURATION > 2) or headCOUNTER < 1:
        exerciseSTART = current_time
    if headCOUNTER < 1:
        exerciseTYPE = "none"
        exerciseAREAnow = "none"
        exerciseAREAprev = "none"
        exerciseRUNNING = False

    print("\nJSON FILE (latest 2 datasets of duration longer than 5 seconds)")
    last_exercises = exercise_data_list[-2:][::-1]
    exercise_data_json = json.dumps(last_exercises, indent=4)
    print(exercise_data_json)

    frame = packet.frame
    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    # Choose font and calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    draw_color_1 = (255, 51, 153)
    draw_color_2 = (204, 0, 204)
    draw_color_3 = (255, 51, 153)

    # Define the size and offset of the rectangle
    rect_width = 500
    rect_height = 500
    bottom_offset = -40  # Offset from the bottom
    left_offset = 0  # Offset to the left from the center
    # Calculate the top-left and bottom-right coordinates of the rectangle
    top_left = ((frame_width - rect_width) // 2 - left_offset, frame_height - rect_height - bottom_offset)
    bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
    # Draw the rectangle on the frame
    cv2.rectangle(frame, top_left, bottom_right, draw_color_1, 4)
    # Text to be added
    text = "Duoballs Area"
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    # Calculate the position for the text
    # Adjusting the text to be centered within the rectangle
    text_x = top_left[0] + (rect_width - text_size[0]) // 2
    text_y = top_left[1] + rect_height - text_size[1] - 60
    # Put the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, draw_color_1, thickness)

    # Define the size and offset of the rectangle
    rect_width = 750
    rect_height = 400
    bottom_offset = 460  # Offset from the bottom
    left_offset = 220  # Offset to the left from the center
    # Calculate the top-left and bottom-right coordinates of the rectangle
    top_left = ((frame_width - rect_width) // 2 - left_offset, frame_height - rect_height - bottom_offset)
    bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
    # Draw the rectangle on the frame
    cv2.rectangle(frame, top_left, bottom_right, draw_color_2, 4)
    # Text to be added
    text = "Triggerpointlever Area"

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    # Calculate the position for the text
    # Adjusting the text to be centered within the rectangle
    text_x = top_left[0] + (rect_width - text_size[0]) // 2
    text_y = top_left[1] + rect_height - text_size[1] - 10
    # Put the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, draw_color_2, thickness)

    if ( exerciseTYPE != "none" ):
        if( exerciseTYPE == "Duoballs" ):
            draw_color_3 = draw_color_1
        else:
            draw_color_3 = draw_color_2
        # Draw the textbox
        cv2.rectangle(frame, (50, 465), (350, 685), (255, 255, 255), -1)
        cv2.rectangle(frame, (50, 465), (350, 685), draw_color_3, 3)
        additional_text = "Type: " + exerciseTYPE
        cv2.putText(frame, additional_text, (70, 505), font, font_scale, draw_color_3, thickness)
        additional_text = "Up & Down: "+str(round(headZ_movement_changes/2))+"x"
        cv2.putText(frame, additional_text, (70, 545), font, font_scale, draw_color_3, thickness)
        formated_sec = f"{exerciseDURATION:.1f}"
        additional_text = "Duration: "+formated_sec+" sec"
        cv2.putText(frame, additional_text, (70, 585), font, font_scale, draw_color_3, thickness)
        additional_text = "Date: "+exerciseSTART.strftime("%Y-%m-%d")
        cv2.putText(frame, additional_text, (70, 625), font, font_scale, draw_color_3, thickness)
        additional_text = "Time: "+exerciseSTART.strftime("%H:%M:%S")
        cv2.putText(frame, additional_text, (70, 665), font, font_scale, draw_color_3, thickness)

    # Print the latest JSON Dataset on the frame
    # Assuming 'frame' is your OpenCV frame
    # Define font, scale, color, and thickness for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    draw_color = (0, 0, 0)
    thickness = 1

    # Check if exercise_data_list is not empty
    if exercise_data_list:
        # Access the last entry
        last_entry = exercise_data_list[-1]

        # Initial text position
        x_json = 960
        y_json = 50
        line_gap = 20  # Gap between lines

        cv2.rectangle(frame, (x_json-10, y_json-20), (x_json+270, y_json+110), (255, 255, 255), -1)

        text = f"[  LATEST JSON DATASET  ]"
        cv2.putText(frame, text, (x_json, y_json), font, font_scale, draw_color, thickness)
        y_json += line_gap  # Move to the next line

        # Loop through the last entry and put each key-value pair on the frame
        for key, value in last_entry.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (x_json, y_json), font, font_scale, draw_color, thickness)
            y_json += line_gap  # Move to the next line

    # Resize the frame
    scale_percent = 0  # percentage of original size
    if ( scale_percent > 0):
        width = int(frame.shape[1] * (100+scale_percent) / 100)
        height = int(frame.shape[0] * (100+scale_percent) / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("TensionTerminator OfficePro T-800 | Ergophysion", resized_frame)
    else:
        cv2.imshow("TensionTerminator OfficePro T-800 | Ergophysion", frame)

    # Display the resized frame

with OakCamera() as oak:
    print("Starting Tension Terminator Live Spatial Detection ...")

    # Setup camera and neural network
    color = oak.create_camera('color')
    nn = oak.create_nn(modelJSON, color, nn_type='yolo', spatial=True)
    oak.visualize(nn.out.main, fps=True, scale=2/3, callback=cb)

    # Start the camera with blocking mode
    oak.start(blocking=True)

    # Stop the camera
    # https://docs.luxonis.com/projects/sdk/en/latest/oak-camera/