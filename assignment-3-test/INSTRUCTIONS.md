# Assignment 3

## PREREQUISITES:
1. Review the Object Detection lectures. Pay special attention to the modules related to the:
- **Rectification Service** (i.e., hard negative mining)
- **Inference Service's Preprocessing module**
- **Deployment** (i.e., Docker)
2. Review Assignment 3. Correct mistakes in your Inference Service, especially those related to NMS and Detector class.
3. Install [ffmpeg](https://ffmpeg.org/download.html) and verify that you can perform UDP streaming on your machine:

Open two terminals on your local machine. In the first terminal, display the streaming video:

```bash
ffplay udp:127.0.0.1:23000
```

Next, in the second terminal, run:

```bash
ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

The first command uses ffplay, which is a simple and portable media player using the FFmpeg libraries. The command is designed to stream (and display) the media stream received from the UDP protocol at the specified IP address (`127.0.0.1`) and port (`23000`). You can use any port number you'd like if it is not in use. The ffmpeg command streams the video file to the specified destination (`udp://127.0.0.1:23000`) at 30 frames per second (`-r 30`).

## REQUIRED LIBRARIES: 
- standard libraries (os, sys, math, itertools, etc.)
- opencv-python (opencv-python-headless for docker implementation)
- numpy
- matplotlib/seaborn
- pandas

> Note: There is an updated **techtrack/requirements.txt** file containing these packages.



## OBJECTIVES: 

You will implement the Rectification Service and package the Inference Service as a containerized Docker service.

**Files to Implement:**
- **techtrack/modules/utils/loss.py**
- **techtrack/modules/rectification/hard_negative_mining.py**
- **techtrack/modules/inference/preprocessing.py**
- **techtrack/app.py**
- **techtrack/README.md**

## TASKS:

Use this repository to make your changes as directed by the instructions and push your changes into your repository. Unit test automatically runs when you push new commits to your repository.

**Task 1:** Review the **loss.py** script. Implement `compute()` class method which calculates the components of a YOLO loss. Follow instructions embedded in the script.

**Task 2:** Review the **hard_negative_mining.py** script. Implement `sample_hard_negatives()` class method that returns the top-N negatives.

**Task 3:** Review the **preprocessing.py** script. Implement the `capture_video()` method to yield every drop-rate'th frame. Keep in mind that, for the Inference Service to perform live detections, it must avoid causing a frame backlog.

> Note: it is useful for you to use the yield keyword instead of the return keyword. You may read this reference. for more information on yield (i.e., generator method).

**Task 4:** Review the **app.py** script. Implement `run()` class method that integrates all three modules in the Inference Service. This service must be able to:

- Capture a stream via the UDP protocol (preprocess module)
- Sequentially detect objects in a frame (model module)
- Filter these detections by applying NMS (nms module)
- Print per-frame detections (i.e, bounding box, class_id, object_score)
- Save the frames with bounding box detections in a directory **storage/detections/**

> NOTE: Do not commit the contents in **storage/detections/**

Finally, write a short _quick start_ tutorial in a **techtrack/README.md** file to outline the instructions needed to run a Docker-packaged Inference Service. 


## SUBMISSION:

Check in all files with your implementation into your provisioned GitHub repository. Push your submission to your GitHub repository and provide your URL when submitting the assignment before the deadline to get credit for this assignment. This assignment is due TWO weeks from the release of this assignment due.