# Tension Terminator OfficePro T-800 | Ergophysion
# code by MCI DiBSE 2021 Group 1.2 "Tschäin"

# shows the use the DepthAI SDK to run different self-trained Yolo NNs and does spatial detection

# run DepthAI SKD
# https://docs.luxonis.com/projects/sdk/en/latest/
# https://docs.luxonis.com/projects/sdk/en/latest/components/nn_component/#nncomponent

import numpy as np
from depthai import NNData
import sys

from depthai_sdk import OakCamera
from depthai_sdk.classes import Detections

# Actual model for Tension Terminator
modeltext = "Yolo 5 MEDIUM 320x320 selected."
modelJSON = "ModelZoo\TTmodel1_yolo5m_img320\TTmodel320v2m.json"

# Check if any arguments were passed
if len(sys.argv) > 1:
    # Iterate over the arguments
    for arg in sys.argv[1:]:
        if arg == "-1":
            modeltext = "Yolo 5 SMALL 640x640 selected."
            modelJSON = "ModelZoo\TTmodel1_yolo5s_img640\TTmodel640v2s.json"
        elif arg == "-2":
            modeltext = "Yolo 5 SMALL 320x320 selected."
            modelJSON = "ModelZoo\TTmodel1_yolo5s_img320\TTmodel320v1.json"
        elif arg == "-3":
            modeltext = "Yolo 8 SMALL 320x320 selected."
            modelJSON = "ModelZoo\TTmodel2_yolov8s_img320\TTmodel2_yolov8s_img320_20231129_212044.json"
        elif arg == "-4":
            modeltext = "Yolo 8 MEDIUM 320x320 selected."
            modelJSON = "ModelZoo\TTmodel2_yolov8m_img320\TTmodel2_yolov8m_img320_20231201_132131.json"

print("\n"+modeltext+"\n")

# decode the data
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


while True:
    with OakCamera() as oak:
        print("Starting Tension Terminator Live Spatial Detection ...")

        color = oak.create_camera('color')
        nn = oak.create_nn(modelJSON, color, nn_type='yolo', spatial=True, decode_fn=decode)
        oak.visualize(nn.out.main, fps=True, scale=1)
        # oak.visualize(nn.out.spatials, fps=2, scale=1/3)
        # oak.visualize(nn.out.passthrough, fps=True)
        # oak.visualize(nn.out.nn_data, fps=True)
        oak.start(blocking=True)