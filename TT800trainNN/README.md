# TT800 Train Neural Network

## Google Colab Notebooks

### Notebook for transfer learning of a YoloV5 model

The **trainTTyolo5.ipynb** Colab Notebook is implementing transfer learning for the **Tension Terminator** based on the **YOLOv5** model. 

The file is structured to guide the user through the entire process, starting from setting up the environment to training the model and eventually exporting the trained weights. Here is a breakdown of its key components:

1. **Setting Global Variables**: The notebook begins by defining several global variables, including the base YOLOv5 model (e.g. `yolov5m`), the number of training epochs (e.g. `70`), the training image size (e.g. `320`), and the project name (`TensionTerminator`). These variables are crucial as they set the foundational parameters for the model training.

2. **Google Drive Integration**: To ensure the trained model and related data are safely stored and accessible, the notebook includes steps to mount Google Drive. This allows for easy backup and retrieval of files associated with the project.

3. **Cloning YOLOv5 Repository**: The notebook includes commands to clone the YOLOv5 repository from GitHub, ensuring the latest version of the model and its dependencies are utilized. After cloning, it navigates to the YOLOv5 directory and installs required packages.

4. **Dependency Installation**: Additional dependencies, like Roboflow, are installed to facilitate the data handling and model training process.

5. **Data Loading from Roboflow**: The notebook utilizes Roboflow, a service for managing and preprocessing datasets for machine learning, to load the required data for the project. It uses a secret key for secure access to the dataset. You can also use your own dataset by uploading it to Google Drive and changing the data path in the `data.yaml` file.

6. **Data Path Corrections**: The paths in the `data.yaml` file are corrected to ensure they point to the appropriate directories within the notebook environment. This step is critical for seamless training and validation processes.

7. **Model Training Command Preparation**: The notebook formulates a command string for training the YOLOv5 model, incorporating the previously set global variables like image size, number of epochs, and base model weights.

8. **Executing Model Training**: The prepared command is executed to start the training process of the YOLOv5 model with the specified parameters and dataset.

9. **Exporting Trained Weights**: Upon completion of the training, the notebook identifies the latest run directory and extracts the best performing model weights (`best.pt`).

10. **File Naming and Google Drive Backup**: The notebook timestamps and renames the extracted weights for clarity and copies them to the mounted Google Drive for safekeeping. This step is crucial for version control and future references.


### Notebook for transfer learning of a YoloV8 model

The **trainTTyolo8.ipynb** Colab Notebook is implementing transfer learning for the **Tension Terminator** based on the **YOLOv8** model.

The Notebook is structured very similar to the Yolo V5 Colab Notebook above.

###
###

## Framegrabbers

### framegrabber2.py
This Python script, designed for the TensionTerminator project, automates the process of extracting frames from video files at specific percentages (5%, 25%, 50%, 75%, and 95%) of their total duration. It leverages OpenCV for video processing, capturing and saving selected frames as PNG files in a dynamically created output directory named using the current timestamp. The script processes all MP4 video files containing 'rgb' in their name, found in the specified `rawdata` directory, and provides a summary of the total number of videos processed and frames captured. This efficient frame extraction tool aids in gathering diverse data samples for training machine learning models, especially in image processing tasks.
```
python3 framegrabber2.py
```

### framegrabber.py
Similar to framegrabber2.py, but this Python script captures frames every two seconds, starting after 5% and ending before 95% of each video's duration to focus on the most relevant content.
```
python3 framegrabber.py
```