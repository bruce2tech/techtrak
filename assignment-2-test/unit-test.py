import os
import tempfile
import unittest
import numpy as np
import cv2
import inspect
from unittest.mock import patch, MagicMock


# Replace these with your actual module paths.
from modules.inference.nms import NMS
from modules.inference.model import Detector
from modules.utils import metrics

###############################################################################
# Tests for Detector (-5pts for each failed test)
###############################################################################
class TestDetector(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for class labels.
        self.temp_class_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
        # Write a few class labels.
        self.classes = ["barcode",
                        "car",
                        "cardboard box",
                        "fire",
                        "forklift",
                        "freight container",
                        "gloves",
                        "helmet",
                        "ladder",
                        "license plate",
                        "person",
                        "qr code",
                        "road sign",
                        "safety vest",
                        "smoke",
                        "traffic cone",
                        "traffic light",
                        "truck",
                        "van",
                        "wood pallet"]
        self.temp_class_file.write("\n".join(self.classes))
        self.temp_class_file.flush()
        self.temp_class_file.close()

    def tearDown(self):
        os.unlink(self.temp_class_file.name)

    def test_predict_with_empty_frame_raises_error(self):
        """
        Test that passing an empty frame to predict raises an error.
        
        Code used:
            dummy_frame = np.empty((0, 0, 3), dtype=np.uint8)
            with patch("cv2.dnn.readNet") as mock_readNet:
                dummy_net = MagicMock()
                dummy_net.getLayerNames.return_value = []
                dummy_net.forward.return_value = []
                mock_readNet.return_value = dummy_net
                detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                                      "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                                      self.temp_class_file.name, score_threshold=0.5)
                detector.predict(dummy_frame)
        """
        code_used = TestDetector.test_predict_with_empty_frame_raises_error.__doc__
        dummy_frame = np.empty((0, 0, 3), dtype=np.uint8)
        with patch("cv2.dnn.readNet") as mock_readNet:
            dummy_net = MagicMock()
            dummy_net.getLayerNames.return_value = []
            dummy_net.forward.return_value = []
            mock_readNet.return_value = dummy_net

            detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                                  "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                                  self.temp_class_file.name, score_threshold=0.5)
            with self.assertRaises(Exception, msg=f"\nTest: test_predict_with_empty_frame_raises_error\nFunction: predict()\nError: An empty frame must raise an error.\nCode used:\n{code_used}\n"):
                detector.predict(dummy_frame)

    def test_post_process_filters_and_converts_detections(self):
        """
        Test that post_process correctly converts bounding boxes from center-based
        to top-left corner format and filters detections based on score_threshold.
        
        Code used:
            detector.img_width = 400
            detector.img_height = 300
            detection1 = np.array([0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.8, 0.05])
            detection2 = np.array([0.3, 0.3, 0.1, 0.1, 0.9, 0.2, 0.1, 0.3])
            predict_output = [np.array([detection1, detection2])]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)
        """
        code_used = TestDetector.test_post_process_filters_and_converts_detections.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection1 = np.array([0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.8, 0.05])
        detection2 = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3])
        predict_output = [np.array([detection1, detection2])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output)

        print(class_scores)

        expected_bbox = [160, 120, 80, 60]
        expected_class_id = 1  # argmax of [0.1, 0.8, 0.05] is index 1.
        expected_confidence = 0.8
        expected_class_scores = detection1[5:]

        self.assertEqual(len(bboxes), 1,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected exactly 1 bounding box, got {}.\nCode used:\n{}\n******\n".format(len(bboxes), code_used))
        self.assertEqual(len(class_ids), 1,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected exactly 1 class ID, got {}.\nCode used:\n{}\n******\n".format(len(class_ids), code_used))
        self.assertEqual(len(confidence_scores), 1,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected exactly 1 confidence score, got {}.\nCode used:\n{}\n******\n".format(len(confidence_scores), code_used))
        self.assertEqual(len(class_scores), 1,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected exactly 1 set of class scores, got {}.\nCode used:\n{}\n******\n".format(len(class_scores), code_used))

        self.assertEqual(bboxes[0], expected_bbox,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected bounding box {} but got {}.\nCode used:\n{}\n******\n".format(expected_bbox, bboxes[0], code_used))
        self.assertEqual(class_ids[0], expected_class_id,
                         "\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                         "Error: Expected class ID {} but got {}.\nCode used:\n{}\n******\n".format(expected_class_id, class_ids[0], code_used))
        self.assertAlmostEqual(confidence_scores[0], expected_confidence,
                               msg="\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                                   "Error: Expected confidence score {} but got {}.\nCode used:\n{}\n******\n".format(expected_confidence, confidence_scores[0], code_used))
        np.testing.assert_array_equal(class_scores[0], expected_class_scores,
                                      err_msg="\n******\nTest: test_post_process_filters_and_converts_detections\nFunction: post_process()\n"
                                              "Error: The class scores do not match expected values.\nCode used:\n" + code_used + "\n******\n")

    def test_post_process_detection_equal_threshold(self):
        """
        Test that a detection with confidence exactly equal to the threshold
        is filtered out (since the condition is strictly greater than the threshold).
        
        Code used:
            detector.img_width = 400
            detector.img_height = 300
            detection = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 0.1, 0.5, 0.05])
            predict_output = [np.array([detection])]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output)
        """
        code_used = TestDetector.test_post_process_detection_equal_threshold.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 0.1, 0.5, 0.05])
        predict_output = [np.array([detection])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output)
        self.assertGreaterEqual(bboxes, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no bounding boxes when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertGreaterEqual(class_ids, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no class IDs when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertGreaterEqual(confidence_scores, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no confidence scores when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertGreaterEqual(class_scores, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no class scores when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))

    def test_post_process_multiple_outputs(self):
        """
        Test that post_process correctly aggregates detections when multiple outputs
        are provided.
        
        Code used:
            detector.img_width = 400
            detector.img_height = 300
            detection1 = np.array([0.4, 0.4, 0.2, 0.2, 0.0, 0.05, 0.7, 0.1])
            detection2 = np.array([0.6, 0.6, 0.1, 0.1, 0.0, 0.2, 0.6, 0.1])
            predict_output = [np.array([detection1]), np.array([detection2])]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output)
        """
        code_used = TestDetector.test_post_process_multiple_outputs.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection1 = np.array([0.4, 0.4, 0.2, 0.2, 0.9, 0.05, 0.7, 0.1])
        detection2 = np.array([0.6, 0.6, 0.1, 0.1, 0.9, 0.2, 0.6, 0.1])
        predict_output = [np.array([detection1]), np.array([detection2])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output)

        expected_bbox1 = [120, 90, 80, 60]   # detection1
        expected_bbox2 = [220, 165, 40, 30]  # detection2

        self.assertEqual(len(bboxes), 2,
                         "\n******\nTest: test_post_process_multiple_outputs\nFunction: post_process()\n"
                         "Error: Expected 2 bounding boxes but got {}.\nCode used:\n{}\n******\n".format(len(bboxes), code_used))
        self.assertEqual(bboxes[0], expected_bbox1,
                         "\n******\nTest: test_post_process_multiple_outputs\nFunction: post_process()\n"
                         "Error: Expected bounding box {} for detection1 but got {}.\nCode used:\n{}\n******\n".format(expected_bbox1, bboxes[0], code_used))
        self.assertEqual(bboxes[1], expected_bbox2,
                         "\n******\nTest: test_post_process_multiple_outputs\nFunction: post_process()\n"
                         "Error: Expected bounding box {} for detection2 but got {}.\nCode used:\n{}\n******\n".format(expected_bbox2, bboxes[1], code_used))
        self.assertEqual(class_ids, [1, 1],
                         "\n******\nTest: test_post_process_multiple_outputs\nFunction: post_process()\n"
                         "Error: Expected class IDs [1, 1] but got {}.\nCode used:\n{}\n******\n".format(class_ids, code_used))
        self.assertAlmostEqual(confidence_scores[0], 0.7,
                               msg="\n******\nTest: test_post_process_multiple_outputs\nFunction: post_process()\n"
                                   "Error: Expected confidence score 0.7 for detection1 but got {}.\nCode used:\n{}\n******\n".format(confidence_scores[0], code_used))

    def test_detector_with_empty_class_file(self):
        """
        Test that when the class file is empty, the detector's classes list is empty.
        
        Code used:
            Create an empty temporary file and pass its name to Detector.
        """
        code_used = TestDetector.test_detector_with_empty_class_file.__doc__
        with tempfile.NamedTemporaryFile(delete=False, mode="w+t") as empty_file:
            empty_file_name = empty_file.name
        try:
            detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                                "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                                empty_file_name, score_threshold=0.5)
            self.assertEqual(detector.classes, [],
                             "\n******\nTest: test_detector_with_empty_class_file\nFunction: __init__()\n"
                             "Error: Expected detector.classes to be empty when class file is empty.\nCode used:\n{}\n******\n".format(code_used))
        finally:
            os.unlink(empty_file_name)

###############################################################################
# Tests for NMS (-5pts for each failed test)
###############################################################################
class TestNMS(unittest.TestCase):
    @patch("cv2.dnn.NMSBoxes")
    def test_filter_with_valid_indices(self, mock_nms):
        """
        We need to verify that when NMS returns valid indices, the filter method 
        correctly maps them to the corresponding bounding boxes, class IDs, scores, 
        and class-specific scores.
        
        Code used:
            bboxes = [[10,10,100,100], [20,20,80,80], [15,15,90,90], [200,200,50,50]]
            class_ids = [0, 1, 0, 2]
            scores = [0.9, 0.75, 0.85, 0.95]
            class_scores = [0.8, 0.6, 0.7, 0.9]
            mock_nms.return_value = np.array([[0], [2]])
            filtered = nms_instance.filter(bboxes, class_ids, scores, class_scores)
        """
        code_used = TestNMS.test_filter_with_valid_indices.__doc__
        bboxes = [
            [10, 10, 100, 100],
            [20, 20, 80, 80],
            [15, 15, 90, 90],
            [200, 200, 50, 50],
        ]
        class_ids = [0, 1, 0, 2]
        scores = [0.9, 0.75, 0.85, 0.95]
        class_scores = [0.8, 0.6, 0.7, 0.9]

        mock_nms.return_value = np.array([[0], [2]])

        nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        filtered = nms_instance.filter(bboxes, class_ids, scores, class_scores)

        expected_bboxes = [bboxes[0], bboxes[2]]
        expected_class_ids = [class_ids[0], class_ids[2]]
        expected_scores = [scores[0], scores[2]]
        expected_class_scores = [class_scores[0], class_scores[2]]

        self.assertEqual(
            {tuple(x) for x in filtered[0]}, {tuple(x) for x in expected_bboxes},
            f"""\nTest: test_filter_with_valid_indices\nFunction: filter()\nError: Filtered bounding boxes do not match expected values.\nCode used:\n{code_used}\nExpected filtered_bboxes = {expected_bboxes}\nBut got: {filtered[0]}\n"""
        )
        self.assertEqual(
            set(filtered[1]), set(expected_class_ids),
            f"""\nTest: test_filter_with_valid_indices\nFunction: filter()\nError: Filtered class IDs do not match expected values.\nCode used:\n{code_used}\nExpected class IDs = {expected_class_ids}\nBut got: {filtered[1]}\n"""
        )
        for a, b in zip(sorted(filtered[2]), sorted(expected_scores)):
            self.assertAlmostEqual(a, b, places=2,
                                    msg=f"""\nTest: test_filter_with_valid_indices\nFunction: filter()\nError: Detection scores mismatch.\nCode used:\n{code_used}\nExpected scores = {expected_scores}\nBut got: {filtered[2]}\nExpected score {b:.3f} but got {a:.3f}.\n""")
        for a, b in zip(sorted(filtered[3]), sorted(expected_class_scores)):
            self.assertAlmostEqual(a, b, places=2,
                                    msg=f"""\nTest: test_filter_with_valid_indices\nFunction: filter()\nError: Class-specific scores mismatch.\nCode used:\n{code_used}\nExpected class scores = {expected_class_scores}\nBut got: {filtered[3]}\nExpected class score {b:.3f} but got {a:.3f}.\n""")

    @patch("cv2.dnn.NMSBoxes")
    def test_filter_with_empty_indices(self, mock_nms):
        """
        Verify that if cv2.dnn.NMSBoxes returns no indices, the filter method returns empty lists.
        
        Code used:
            bboxes = [[10,10,100,100], [20,20,80,80]]
            class_ids = [0, 1]
            scores = [0.3, 0.4]
            class_scores = [0.2, 0.3]
            mock_nms.return_value = []
            result = nms_instance.filter(bboxes, class_ids, scores, class_scores)
        """
        code_used = TestNMS.test_filter_with_empty_indices.__doc__
        bboxes = [[10, 10, 100, 100], [20, 20, 80, 80]]
        class_ids = [0, 1]
        scores = [0.3, 0.4]
        class_scores = [0.2, 0.3]

        mock_nms.return_value = []
        nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        result = nms_instance.filter(bboxes, class_ids, scores, class_scores)
        self.assertEqual(result, ([], [], [], []),
                         f"""\nTest: test_filter_with_empty_indices\nFunction: filter()\nError: When NMSBoxes returns no indices, expected output ([], [], [], []), but got {result}.\nCode used:\n{code_used}\n""")

    @patch("cv2.dnn.NMSBoxes")
    def test_filter_with_single_index(self, mock_nms):
        """
        Verify that when a single index is returned, the filter method correctly returns that detection.
        
        Code used:
            bboxes = [[10,10,50,50], [12,12,48,48]]
            class_ids = [0, 0]
            scores = [0.95, 0.94]
            class_scores = [0.9, 0.88]
            mock_nms.return_value = np.array([[0]])
            filtered = nms_instance.filter(bboxes, class_ids, scores, class_scores)
        """
        code_used = TestNMS.test_filter_with_single_index.__doc__
        bboxes = [[10, 10, 50, 50], [12, 12, 48, 48]]
        class_ids = [0, 0]
        scores = [0.95, 0.94]
        class_scores = [0.9, 0.88]

        mock_nms.return_value = np.array([[0]])
        nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        filtered = nms_instance.filter(bboxes, class_ids, scores, class_scores)
        self.assertEqual(filtered[0], [bboxes[0]],
                         f"""\nTest: test_filter_with_single_index\nFunction: filter()\nError: Expected bounding box {bboxes[0]} but got {filtered[0]}.\nCode used:\n{code_used}\n""")
        self.assertEqual(filtered[1], [class_ids[0]],
                         f"""\nTest: test_filter_with_single_index\nFunction: filter()\nError: Expected class ID {class_ids[0]} but got {filtered[1]}.\nCode used:\n{code_used}\n""")
        for a, b in zip(sorted(filtered[2]), sorted([scores[0]])):
            self.assertAlmostEqual(a, b, places=2,
                                    msg=f"""\nTest: test_filter_with_single_index\nFunction: filter()\nError: Expected score {scores[0]:.3f} but got {a:.3f}.\nCode used:\n{code_used}\n""")
        for a, b in zip(sorted(filtered[3]), sorted([class_scores[0]])):
            self.assertAlmostEqual(a, b, places=2,
                                    msg=f"""\nTest: test_filter_with_single_index\nFunction: filter()\nError: Expected class-specific score {class_scores[0]:.3f} but got {a:.3f}.\nCode used:\n{code_used}\n""")

    def test_filter_with_empty_input_lists(self):
        """
        Verify that empty input lists result in empty output lists.
        
        Code used:
            result = nms_instance.filter([], [], [], [])
        Expected output: ([], [], [], []).
        """
        code_used = TestNMS.test_filter_with_empty_input_lists.__doc__
        nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        self.assertEqual(nms_instance.filter([], [], [], []), ([], [], [], []),
                         f"""\nTest: test_filter_with_empty_input_lists\nFunction: filter()\nError: Empty input lists should yield empty output lists.\nCode used:\n{code_used}\n""")


###############################################################################
# Tests for mAP (-5pts for each failed test)
###############################################################################
class TestMAP(unittest.TestCase):
    def setUp(self):
        # Set up for a 20-class object detection problem
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [50, 50, 30, 30]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 2, 3],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.90, 0.85, 0.30],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.85, 0.80, 0.75, 0.30],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:, None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]
        # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold  # IoU threshold for a valid match
            # Optionally, eval_type="objectness" could be passed if needed.
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        self.baseline_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

    def test_mAP_values_between_zero_and_one(self):
        error_message = "\n\n>>> Value Error: mAP should be between zero and one (e.g, [0, 1])."

        self.assertGreaterEqual(self.baseline_map, 0,  msg=error_message)
        self.assertLessEqual(self.baseline_map, 1, msg=error_message)

    def test_high_threshold_decrease_map_score(self):
        # Set up for a 20-class object detection problem
        code_used = inspect.getsource(TestMAP.test_high_threshold_decrease_map_score).replace("\n","\n>>>")

        num_classes = 20
        map_iou_threshold = 0.8

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [50, 50, 30, 30]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 2, 3],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.90, 0.85, 0.30],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.85, 0.80, 0.75, 0.30],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]

        # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold  # IoU threshold for a valid match
            # Optionally, eval_type="objectness" could be passed if needed.
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))

    def test_low_obj_scores(self):
        code_used = inspect.getsource(TestMAP.test_low_obj_scores).replace("\n","\n>>>")
        # Set up for a 20-class object detection problem
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [50, 50, 30, 30]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 2, 3],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.90, 0.85, 0.30],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.00, 0.00, 0.00, 0.00],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]

       # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold,  # IoU threshold for a valid match
            eval_type="class_scores"
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))

    def test_duplicate_bboxes(self):
        # Set up for a 20-class object detection problem
        code_used = inspect.getsource(TestMAP.test_duplicate_bboxes).replace("\n","\n>>>")
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [50, 50, 30, 30]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 0, 1, 2, 3],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.95, 0.90, 0.85, 0.30],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.85, 0.85, 0.80, 0.75, 0.30],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]

       # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold,  # IoU threshold for a valid match
            eval_type="class_scores"
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))

    def test_misclassification(self):
        code_used = inspect.getsource(TestMAP.test_misclassification).replace("\n","\n>>>")
        # Set up for a 20-class object detection problem
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [50, 50, 30, 30]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 3, 3],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.90, 0.85, 0.99],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.85, 0.80, 0.75, 0.99],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        # cls_scores = [
        #     np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
        #     for class_list, score_list in zip(classes, dummy_max_cls_scores)
        # ]

        
        cls_scores =[np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
                for class_list, score_list in zip(classes, dummy_max_cls_scores)
            ]

       # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold,  # IoU threshold for a valid match
            eval_type="class_scores"
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))

    def test_false_negatives(self):
        code_used = inspect.getsource(TestMAP.test_false_negatives).replace("\n","\n>>>")
        # Set up for a 20-class object detection problem
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------

        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # We create predictions that try to detect the GT objects and also include a few extra
        # or misclassified detections to cover all 20 classes.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            #  - Three predictions that ideally match the GT objects (classes 0, 1, 2)
            #  - One extra detection with class 3 (which is not present in GT for image 1)
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70]],
            
            # Image 2 predictions:
            #  - Three predictions for GT objects (should be classes 3, 4, 5) but one is misclassified
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            #  - Two correct predictions for classes 12 and 13
            #  - One misclassified detection (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 2],    # Image 1: extra detection with class 3 (false positive)
            [3, 4, 6],       # Image 2: last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: all correct
            [9, 10, 11],     # Image 4: all correct
            [12, 13, 15],    # Image 5: last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: all correct
        ]

        scores = [
            [0.95, 0.90, 0.85],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These scores are used to simulate
        # the output from a classification head. We create a one-hot encoded score vector
        # (of length num_classes) for each detection and multiply by the dummy score.
        dummy_max_cls_scores = [
            [0.85, 0.80, 0.75],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:,None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]

       # ------------------------------------------------------------------------------
        # Evaluation
        #
        # Use your metrics module functions to evaluate detections and compute mAP.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold,  # IoU threshold for a valid match
            eval_type="class_scores"
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))

    def test_false_positives(self):
        code_used = inspect.getsource(TestMAP.test_false_positives).replace("\n","\n>>>")
        # Set up for a 20-class object detection problem
        num_classes = 20
        map_iou_threshold = 0.5

        # ------------------------------------------------------------------------------
        # Ground Truth (gt_boxes and gt_classes)
        #
        # We create 6 images with the following distribution:
        #   - Image 1: GT classes: [0, 1, 2]
        #   - Image 2: GT classes: [3, 4, 5]
        #   - Image 3: GT classes: [6, 7, 8]
        #   - Image 4: GT classes: [9, 10, 11]
        #   - Image 5: GT classes: [12, 13, 14]
        #   - Image 6: GT classes: [15, 16, 17, 18, 19]
        # ------------------------------------------------------------------------------
        gt_boxes = [
            # Image 1
            [[33, 117, 259, 396], [362, 161, 259, 362], [100, 100, 50, 80]],
            # Image 2
            [[163, 29, 301, 553], [400, 200, 100, 150], [50, 50, 80, 120]],
            # Image 3
            [[10, 10, 60, 60], [100, 50, 120, 180], [200, 200, 70, 70]],
            # Image 4
            [[250, 250, 80, 100], [300, 300, 60, 80], [400, 400, 90, 90]],
            # Image 5
            [[50, 300, 100, 120], [200, 350, 150, 150], [350, 50, 100, 100]],
            # Image 6
            [[400, 50, 80, 80], [420, 60, 90, 90], [440, 70, 100, 100],
            [460, 80, 110, 110], [480, 90, 120, 120]]
        ]

        gt_classes = [
            [0, 1, 2],       # Image 1
            [3, 4, 5],       # Image 2
            [6, 7, 8],       # Image 3
            [9, 10, 11],     # Image 4
            [12, 13, 14],    # Image 5
            [15, 16, 17, 18, 19]  # Image 6
        ]

        # ------------------------------------------------------------------------------
        # Predictions (boxes, classes, scores, dummy_max_cls_scores)
        #
        # For the false positive test, we purposely add an extra detection for one image
        # that does not correspond to any ground truth box.
        # ------------------------------------------------------------------------------
        boxes = [
            # Image 1 predictions:
            # Three detections ideally match the GT objects (classes 0, 1, 2),
            # plus an extra detection with a random location and class 3.
            [[30, 187, 253, 276], [360, 150, 270, 380], [110, 110, 45, 70], [500, 500, 50, 50]],
            
            # Image 2 predictions:
            [[160, 25, 300, 550], [390, 210, 110, 140], [55, 55, 80, 120]],
            
            # Image 3 predictions: correct detections for classes 6, 7, 8
            [[12, 12, 58, 58], [105, 45, 115, 190], [205, 205, 65, 65]],
            
            # Image 4 predictions: correct detections for classes 9, 10, 11
            [[248, 248, 85, 105], [310, 310, 55, 75], [405, 405, 85, 85]],
            
            # Image 5 predictions:
            # Two correct detections for classes 12 and 13, plus one misclassified detection
            # (should be 14 but predicted as 15)
            [[55, 305, 95, 115], [195, 345, 160, 160], [345, 45, 105, 105]],
            
            # Image 6 predictions: correct detections for classes 15, 16, 17, 18, 19
            [[402, 52, 78, 78], [425, 65, 85, 85], [435, 75, 95, 95], [465, 85, 108, 108], [478, 92, 115, 115]]
        ]

        classes = [
            [0, 1, 2, 3],    # Image 1: Extra detection (class 3) is a false positive since no GT of class 3 exists in Image 1.
            [3, 4, 6],       # Image 2: Last prediction misclassified (should be 5, but predicted as 6)
            [6, 7, 8],       # Image 3: All correct
            [9, 10, 11],     # Image 4: All correct
            [12, 13, 15],    # Image 5: Last detection misclassified (should be 14 but predicted as 15)
            [15, 16, 17, 18, 19]  # Image 6: All correct
        ]

        scores = [
            [0.95, 0.90, 0.85, 0.99],  # Image 1 scores
            [0.92, 0.88, 0.40],         # Image 2 scores
            [0.93, 0.89, 0.87],         # Image 3 scores
            [0.94, 0.91, 0.88],         # Image 4 scores
            [0.90, 0.86, 0.50],         # Image 5 scores
            [0.95, 0.93, 0.91, 0.88, 0.85]  # Image 6 scores
        ]

        # Dummy classification scores for each detection. These simulate the output from a classification head.
        dummy_max_cls_scores = [
            [0.85, 0.80, 0.75, 0.99],  # Image 1 dummy scores
            [0.83, 0.78, 0.35],         # Image 2 dummy scores
            [0.88, 0.84, 0.80],         # Image 3 dummy scores
            [0.87, 0.82, 0.79],         # Image 4 dummy scores
            [0.86, 0.81, 0.45],         # Image 5 dummy scores
            [0.90, 0.88, 0.85, 0.82, 0.80]  # Image 6 dummy scores
        ]

        # Convert the class labels and dummy scores to one-hot encoded score vectors.
        cls_scores = [
            np.eye(num_classes)[np.array(class_list)] * np.array(score_list)[:, None]
            for class_list, score_list in zip(classes, dummy_max_cls_scores)
        ]

        # ------------------------------------------------------------------------------
        # Evaluation: Compute mAP using the provided metrics functions.
        # ------------------------------------------------------------------------------
        y_true, pred_scores = metrics.evaluate_detections(
            boxes,         # Detected bounding boxes per image
            classes,       # Detected class labels per image
            scores,        # Detection confidence scores per image
            cls_scores,    # Classification score vectors per image
            gt_boxes,      # Ground truth bounding boxes per image
            gt_classes,    # Ground truth class labels per image
            map_iou_threshold  # IoU threshold for a valid match
            # Optionally, eval_type="objectness" can be passed if needed.
        )

        precision, recall, thresholds = metrics.calculate_precision_recall_curve(
            y_true,         # True labels from the evaluation
            pred_scores,    # Predicted scores from the evaluation
            num_classes=num_classes
        )

        precision_recall_points = {
            class_index: list(zip(recall[class_index], precision[class_index]))
            for class_index in range(num_classes)
        }

        test_map = metrics.calculate_map_x_point_interpolated(precision_recall_points, num_classes)

        self.assertGreater(self.baseline_map, test_map, msg="\nTest code used:\n>>>{}\n".format(code_used))


if __name__ == '__main__':
    unittest.main()
