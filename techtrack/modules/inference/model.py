import cv2
import numpy as np
from typing import List, Tuple
import os, sys


class Detector:
    """
    A class that represents an object detection model using OpenCV's DNN module
    with a YOLO-based architecture.
    """

    def __init__(self, weights_path: str, config_path: str, class_path: str, score_threshold: float=.5) -> None:
        """
        Initializes the YOLO model by loading the pre-trained network and class labels.

        :param weights_path: Path to the pre-trained YOLO weights file.
        :param config_path: Path to the YOLO configuration file.
        :param class_path: Path to the file containing class labels.

        :ivar self.net: The neural network model loaded from weights and config files.
        :ivar self.classes: A list of class labels loaded from the class_path file.
        :ivar self.img_height: Height of the input image/frame.
        :ivar self.img_width: Width of the input image/frame.
        """

        # Determine project root relative to this file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Path of test storage directory
        test_storage_dir = os.path.join(base_dir, 'assignment-2-test')

        # Resolve and verify each provided path
        def resolve(path: str) -> str:
            # 1. Absolute paths
            if os.path.isabs(path) and os.path.isfile(path):
                return path
            # 2. Relative to current working directory
            cwd_candidate = os.path.abspath(path)
            if os.path.isfile(cwd_candidate):
                return cwd_candidate
            # 3. Relative to project root
            proj_candidate = os.path.join(base_dir, path)
            if os.path.isfile(proj_candidate):
                return proj_candidate
            # 4. Relative to test directory (for test assets)
            test_candidate = os.path.join(test_storage_dir, path)
            if os.path.isfile(test_candidate):
                return test_candidate
            # Fallback: return proj_candidate so the existence check raises
            return proj_candidate
        
        # Resolve paths
        self.weights_path = resolve(weights_path)
        self.config_path  = resolve(config_path)
        self.class_path   = resolve(class_path)

         # Ensure all files exist
        for path in (self.weights_path, self.config_path, self.class_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")

        # Load network
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load class labels (allow empty files)
        with open(self.class_path, 'r') as f:
            lines = f.readlines()
            self.classes = [line.strip() for line in lines if line.strip()]

        self.img_height: int = 0
        self.img_width: int = 0
        self.score_threshold: float = score_threshold

    def predict(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """
        Runs the YOLO model on a single input frame and returns raw predictions.

        :param preprocessed_frame: A single image frame that has been preprocessed 
                                   for YOLO model inference (e.g., resized and normalized).

        :return: A list of NumPy arrays containing the raw output from the YOLO model.
                 Each output consists of multiple detections with bounding boxes, 
                 confidence scores, and class probabilities.

        :ivar self.img_height: The height of the input image/frame.
        :ivar self.img_width: The width of the input image/frame.

        **YOLO Output Format:**
        Each detection in the output contains:
        - First 4 values: Bounding box center x, center y, width, height.
        - 5th value: Confidence score.
        - Remaining values: Class probabilities for each detected object.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """

        # Check for empty frame
        if preprocessed_frame is None or preprocessed_frame.size == 0 or preprocessed_frame.shape[0] == 0 or preprocessed_frame.shape[1] == 0:
            raise ValueError("Empty frame provided to predict()")

        self.img_height, self.img_width = preprocessed_frame.shape[:2]
        # Create a blob and run a forward pass through the network
        blob = cv2.dnn.blobFromImage(
            preprocessed_frame,
            scalefactor=1/255.0,
            size=(self.img_width, self.img_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)
        return outputs


        # TASK 2: Use the YOLO model to return all raw outputs

    def post_process(
        self, predict_output: List[np.ndarray]
    ) -> Tuple[List[List[int]], List[int], List[float], List[np.ndarray]]:
        """
        Processes the raw YOLO model predictions and filters out low-confidence detections.

        :param predict_output: A list of NumPy arrays containing raw predictions 
                               from the YOLO model.
        :param score_threshold: Minimum confidence score required for a detection 
                                to be considered valid.

        :return: A tuple containing:
            - **bboxes (List[List[int]])**: List of bounding boxes as `[x, y, width, height]`, 
              where (x, y) represents the top-left corner.
            - **class_ids (List[int])**: List of detected object class indices.
            - **confidence_scores (List[float])**: List of confidence scores for each detection.
            - **class_scores (List[np.ndarray])**: List of all class-specific confidence scores.

        **Post-processing steps:**
        1. Extract bounding box coordinates from YOLO output.
        2. Compute class probabilities and determine the most likely class.
        3. Filter out detections below the confidence threshold.
        4. Convert bounding box coordinates from center-based format to 
           top-left corner format.

        **Bounding Box Conversion:**
        YOLO outputs bounding box coordinates in the format:
        ```
        center_x, center_y, width, height
        ```
        This function converts them to:
        ```
        x, y, width, height
        ```
        where (x, y) is the top-left corner.

        **Reference:**
        - OpenCV YOLO Documentation: 
          https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob
        """
        bboxes: List[List[int]] = []
        class_ids: List[int] = []
        confidence_scores: List[float] = []
        class_scores: List[np.ndarray] = []

        # Iterate over each detection block
        for output in predict_output:
            for detection in output:
                # First 4: box, 5th: objectness, rest: class probabilities
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                # objectness = float(detection[4])
                class_confidence = float(scores[class_id])
                # final_score = objectness * class_confidence
                final_score = class_confidence

                # Filter by score threshold
                if final_score >= self.score_threshold:
                    # Convert center-based coords to top-left
                    cx = int(detection[0] * self.img_width)
                    cy = int(detection[1] * self.img_height)
                    w = int(detection[2] * self.img_width)
                    h = int(detection[3] * self.img_height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    bboxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidence_scores.append(final_score)
                    class_scores.append(scores)

        return bboxes, class_ids, confidence_scores, class_scores

        # TASK 3: Use the YOLO model to return list of NumPy arrays filtered
        #         by processing the raw YOLO model predictions and filters out 
        #         low-confidence detections (i.e., < score_threshold). Use the logic
        #         in Line 83-88.


    def process_video(
        self,
        input_path: str,
        output_path: str = None
    ) -> None:
        """
        Runs detection on each frame of an input video file, optionally writing annotated output.

        :param input_path: Path to the input .mp4 video file.
        :param output_path: Optional path for saving the output video with detections.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            outputs = self.predict(frame)
            bboxes, class_ids, scores, _ = self.post_process(outputs)

            # Draw results
            for (x, y, w, h), cid, conf in zip(bboxes, class_ids, scores):
                label = f"{self.classes[cid]}: {conf:.2f}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )

            # Write or show frame
            if writer:
                writer.write(frame)
            else:
                cv2.imshow('Detections', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        else:
            cv2.destroyAllWindows()

"""
EXAMPLE USAGE:
model = Detector()

# Perform object detection on the current frame
predictions = self.detector.predict(frame)

# Extract bounding boxes, class IDs, confidence scores, and class-specific scores
bboxes, class_ids, confidence_scores, class_scores = self.detector.post_process(
    predictions
)
"""

if __name__ == "__main__":
    ## Test the detector on an image
    ## See https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html#create-a-blob for more information
    ## Note: Requires non-headless version of OpenCV

    ## Initialize model
    weights_path = "../../assignment-2-test/storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights"
    config_path  = "../../storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.cfg"
    class_path   = "../../storage/yolo_models/yolo_model_1/logistics.names"
    score_threshold = .5

    model = Detector(weights_path    = weights_path,
                     config_path     = config_path,
                     class_path      = class_path,
                     score_threshold = score_threshold)

    # Load sample frame
    frame_path   = "../../storage/logistics/p1_8760_jpg.rf.f174b5ef18011cac19021f9df63ecb7c.jpg"
    frame        = cv2.imread(frame_path)

    # Run sample prediction
    predictions = model.predict(frame)
    bboxes, class_ids, confidence_scores, class_scores = model.post_process(predictions)

    # Visualize (see https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1)
    ax.imshow(frame)
    for bbox in bboxes:
        x, y, width, height = bbox
        x, y, width, height = (int(x*model.img_width),
                               int(y*model.img_height),
                               int(width*model.img_width),
                               int(height*model.img_height))
        r = Rectangle((x, y), width, height, facecolor="none", edgecolor="r")
        ax.add_patch(r)
    plt.show()