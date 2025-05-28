import cv2
import numpy as np
from typing import List, Tuple


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
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load class labels
        with open(class_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.img_height: int = 0
        self.img_width: int = 0

        self.score_threshold = score_threshold

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
        self.img_height, self.img_width = preprocessed_frame.shape[:2]

        # TASK 2: Use the YOLO model to return all raw outputs

        blob = cv2.dnn.blobFromImage(preprocessed_frame,
                                     scalefactor = 1/255.,
                                     size        = (self.img_height, self.img_width),
                                     swapRB      = True,
                                     crop        = False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        # Return model outputs:
        return outputs

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
        
        # TASK 3: Use the YOLO model to return list of NumPy arrays filtered
        #         by processing the raw YOLO model predictions and filters out 
        #         low-confidence detections (i.e., < score_threshold). Use the logic
        #         in Line 83-88.
        raw_boxes, class_ids, confidence_scores, class_scores = [], [], [], []

        # Discard low-threshold detections
        for feature_maps in predict_output:
            for detection in feature_maps:
                if detection[4] > self.score_threshold:
                    raw_boxes.append(detection[:4])
                    confidence_scores.append(detection[4])
                    class_scores.append(detection[5:])
                    class_ids.append(np.argmax(class_scores[-1]))

        # Convert box coordinates
        bboxes = []
        for box in raw_boxes:
            center_x, center_y, width, height = box
            x = center_x - width  / 2.
            y = center_y - height / 2.
            bboxes.append([x, y, width, height])

        # Return these variables in order:
        return bboxes, class_ids, confidence_scores, class_scores


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
    weights_path = "../../storage/yolo_models/yolo_model_1/yolov4-tiny-logistics_size_416_1.weights"
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