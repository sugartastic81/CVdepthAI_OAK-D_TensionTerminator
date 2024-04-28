import cv2
import os
import datetime

# Function to capture frames at specified percentages of the video duration
def capture_frames(video_path, output_folder, img_counter, folder_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return img_counter, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame numbers for x% of the video
    frame_numbers = [
        int(total_frames * 0.05),
        int(total_frames * 0.25),
        int(total_frames * 0.50),
        int(total_frames * 0.75),
        int(total_frames * 0.95)
    ]

    frames_captured = 0
    for frame_number in frame_numbers:
        # Move to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            output_filename = f'{folder_name}_{img_counter:06}_{frame_number:04}.png'
            cv2.imwrite(os.path.join(output_folder, output_filename), frame)
            img_counter += 1
            frames_captured += 1
        else:
            print(f"Error: Could not read frame {frame_number} from video {video_path}.")

    cap.release()
    return img_counter, frames_captured

print("\n\n************************************")
print("* TensionTerminator Framegrabber 2 *")
print("************************************\n")

# Starting directory
start_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "rawdata")
# Output folder within 'traindata' directory
output_folder = os.path.join(start_dir, f"framesgrabbed_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
os.makedirs(output_folder, exist_ok=True)

# Count total eligible videos
total_videos = sum(1 for _, _, files in os.walk(start_dir) for file in files if file.endswith(".mp4") and 'rgb' in file.lower())
processed_videos = 0
total_frames_grabbed = 0

# Traverse directories and process videos
img_counter = 1
for root, dirs, files in os.walk(start_dir):
    folder_name = os.path.basename(root)
    for file in files:
        if file.endswith(".mp4") and 'rgb' in file.lower():
            video_path = os.path.join(root, file)
            img_counter, frames_captured = capture_frames(video_path, output_folder, img_counter, folder_name)
            processed_videos += 1
            total_frames_grabbed += frames_captured
            percentage_done = (processed_videos / total_videos) * 100
            print(f"{frames_captured} frames grabbed from '{video_path}'\t\tvideo {processed_videos} of {total_videos}\t\t{percentage_done:.1f}% processed")

# Output summary of processed videos and frames
print("\n\n************************************")
print("*              SUMMARY              *")
print("************************************")
print(f"Total videos processed: {processed_videos}")
print(f"Total frames grabbed: {total_frames_grabbed}")
print(f"Output directory: {output_folder}\n\n")
