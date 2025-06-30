# Assignment 2

## PREREQUISITES:
1. Review the Object Detection lectures. Pay special attention to the modules related to the:
- Inference Service (preprocessing, object detection, and non-maximum suppression modules)
- Metrics and object detection Mean Average Precision.
2. Study the OpenCV YOLO tutorial. You are provided the skeleton code to guide you with the model implementation. Each function is provided with its description in the docstrings. 
3. Study the TechTrack data resources. Place these data files in their corresponding directories. Do NOT check the following data in your Github repository.
- storage/logistics: unzipped logistics dataset
- storage/test_videos: unzipped test_videos
- storage/yolo_models: unzipped yolo_model_1 and yolo_model_2


## REQUIRED LIBRARIES:
- standard libraries (os, sys, math, itertools, etc.)
- opencv-python (use opencv-python-headless for Docker implementation)
- numpy

> NOTE: When deploying in Docker, replace opencv-python with opencv-python-headless to avoid GUI-related dependency issues. See requirements.txt file. 

## OBJECTIVES: 

You will implement the Inference Service and Mean Average Precision. 

**Files to Implement:**
- **techtrack/modules/inference/model.py**
- **techtrack/modules/inference/nms.py**
- **techtrack/models/utils/metrics.py**

> NOTE: You may add additional methods, but DO NOT modify import statements without prior instructor approval. Should you decide to add additional arguments to your functions, make sure to provide them default parameters.

## TASKS:

Use this link to fork the TechTrack base repository for this assignment into your personal GitHub account. Please update your current repository as there may be updates to the repository. Make your changes as directed by the instructions and push your changes into your repository. Unit test automatically runs when you push new commits to your repository.

**Task 1:** Review the **model.py** script. Implement the `predict()` method to output ALL the predictions of the YOLO model, and `post_process()` method to filter the predictions of the YOLO model based on a score_threshold.

**Task 2:** Review the **nms.py** script. Complete the `filter()` method to apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes. Only use the numpy package for this task. After this assignment, you can use opencv's implementation.

**Task 3-4:** Review the **metrics.py** script and the runner demonstrating the functions. Detailed implementation is provided in each function's docstrings. Complete the following tasks:

- Evaluate model detections against the ground truth objects using the function evaluate_detections()
- Compute the function `calculate_precision_recall_curve()`

> NOTE: The function `calculate_map_x_point_interpolated()` is implemented for you. For this task, it may be helpful to complete the `calculate_iou()` function if you haven't already.

## SUBMISSION:

Check in all files with your implementation into your provisioned GitHub repository. Push your submission to your GitHub repository and provide your URL when submitting the assignment before the deadline to get credit for this assignment. This assignment is due TWO weeks from the release of this assignment due.