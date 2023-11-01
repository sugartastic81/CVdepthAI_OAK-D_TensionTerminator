# Tension Terminator Test

import cv2
import depthai as dai
import numpy as np

def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono


def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
    #Preset modes for stereo depth
    stereo.PresetMode.HIGH_ACCURACY
    # Closer-in minimum depth, disparity range is doubled. Roughly from 70cm distance to 35cm distance
    stereo.setExtendedDisparity(True)
    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)
    # Better accuracy for longer distance, fractional disparity 32-levels: (is in conflict with extended disparity)
    stereo.setSubpixel(False)
    # Override focal length from calibration. This parameter is in pixel units computed as (baseline * focal_length / pixel_size).
    stereo.setFocalLength(150)

    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)


    print("Stereo Depth config: ", stereo.inputConfig)

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


if __name__ == '__main__':

    mouseX = 0
    mouseY = 640
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # Set XlinkOut for disparity, rectifiedLeft, and rectifiedRight
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)

    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    # Pipeline is defined, now we can connect to the device

    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)

        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255 / 90

        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallback)

        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False

        while True:

            # Get disparity map
            origin_disparity = getFrame(disparityQueue)

            # Colormap disparity for display
            disparity1 = (origin_disparity * disparityMultiplier).astype(np.uint8)
            disparity1 = cv2.applyColorMap(disparity1, cv2.COLORMAP_JET)
            disparity2 = disparity1

            MIN_DISPARITY = 60  # Beispielwert für die minimale Disparität
            MAX_DISPARITY = 500  # Beispielwert für die maximale Disparität
            mask1 = (disparity1 > MIN_DISPARITY) & (disparity1 < MAX_DISPARITY)
            disparity_masked1 = np.where(mask1, disparity1, 0)
            disparity_masked1 = cv2.applyColorMap(disparity_masked1, cv2.COLORMAP_JET)

            MIN_DISPARITY = 30  # Beispielwert für die minimale Disparität
            MAX_DISPARITY = 400  # Beispielwert für die maximale Disparität
            mask2 = (disparity2 > MIN_DISPARITY) & (disparity2 < MAX_DISPARITY)
            disparity_masked2 = np.where(mask2, disparity2, 0)
            disparity_masked2 = cv2.applyColorMap(disparity_masked2, cv2.COLORMAP_JET)


            #disparity_masked1 = cv2.bilateralFilter(disparity_masked1, 9, 75, 75)
            #disparity_masked2 = cv2.bilateralFilter(disparity_masked2, 9, 75, 75)
            #
            # kernel = np.ones((5, 5), np.uint8)
            # disparity = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
            # disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)

            # MIN_DEPTH = 10  # Beispielwert
            # MAX_DEPTH = 200  # Beispielwert
            #
            # mask = (disparity > MIN_DEPTH) & (disparity < MAX_DEPTH)
            # disparity = np.where(mask, disparity, 0)

            # Get left and right rectified frame
            leftFrame = getFrame(rectifiedLeftQueue);
            rightFrame = getFrame(rectifiedRightQueue)

            # Show side by side view
            imOut1 = np.hstack((leftFrame, rightFrame))
            imOut1 = cv2.cvtColor(imOut1, cv2.COLOR_GRAY2RGB)

            # Show overlapping frames
            imOut2 = np.uint8(leftFrame / 2 + rightFrame / 2)
            imOut2 = cv2.cvtColor(imOut2, cv2.COLOR_GRAY2RGB)


            cv2.imshow("Stereo - Side by Side", imOut1)
            cv2.imshow("Stereo - Overlapping", imOut2)
            cv2.imshow("Disparity_original", disparity1)
            cv2.imshow("disparity_masked1", disparity_masked1)
            cv2.imshow("disparity_masked2", disparity_masked2)


            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide

