# grabbes frames from teh mp4 rgb videos recorded by TT in a directory and subdirectories for data preprocessing
# starts after 5% of the video and ends before 95% of the video
# grabs every 2 seconds of the video

import cv2
import os
import datetime


# Function to capture frames from a video file
def capture_frames(video_path, output_folder, img_counter, folder_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return img_counter, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Start and end frames for capturing
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)

    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_captured = 0
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if ret:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            output_filename = f'{folder_name}_{img_counter:06}_{frame_number:04}.png'
            cv2.imwrite(os.path.join(output_folder, output_filename), frame)
            img_counter += 1
            frames_captured += 1

            # Skip to the frame at the next 2-second interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 2 * fps)
        else:
            break

    cap.release()
    return img_counter, frames_captured

print("\n\n**********************************")
print("* TensionTerminator Framegrabber *")
print("**********************************\n")


# Starting directory
start_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rawdata")
# Output folder within 'traindata' directory
output_folder = os.path.join(start_dir, f"framesgrabbed_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
os.makedirs(output_folder, exist_ok=True)

# Variables to keep track of video and frame counts
total_videos = 0
total_frames_grabbed = 0

# Traverse directories and process videos
img_counter = 1
for root, dirs, files in os.walk(start_dir):
    folder_name = os.path.basename(root)
    for file in files:
        if file.endswith(".mp4") and 'rgb' in file.lower():
            video_path = os.path.join(root, file)
            img_counter, frames_captured = capture_frames(video_path, output_folder, img_counter, folder_name)

            if frames_captured > 0:
                print(f"Processed video: {file}")
                print(
                    f"Video length: {cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT) / cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS):.2f} seconds")
                print(f"Frames grabbed: {frames_captured}\n")

                total_videos += 1
                total_frames_grabbed += frames_captured

# Output summary of processed videos and frames
print(f"Total videos processed: {total_videos}")
print(f"Total frames grabbed: {total_frames_grabbed}")
