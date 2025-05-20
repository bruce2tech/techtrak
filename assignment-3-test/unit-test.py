import os
import glob
import tempfile
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Replace these with your actual module paths.
from modules.inference.preprocessing import Preprocessing
from modules.inference.nms import NMS
from modules.inference.model import Detector


###############################################################################
# DummyVideoCapture for Preprocessing Tests
###############################################################################
class DummyVideoCapture:
    """
    A dummy video capture class to simulate cv2.VideoCapture behavior.
    
    If provided with a list of frames, it uses them directly. If provided with a string
    and the string is a path to a directory, it loads all image files from that directory 
    (sorted alphabetically) as frames.
    """
    def __init__(self, source, open_success=True):
        self.index = 0
        self.open_success = open_success

        if isinstance(source, list):
            self.frames = source
        elif isinstance(source, str) and os.path.isdir(source):
            files = sorted(glob.glob(os.path.join(source, "*")))
            self.frames = []
            for f in files:
                img = cv2.imread(f)
                if img is not None:
                    self.frames.append(img)
        else:
            self.frames = []

    def isOpened(self):
        return self.open_success

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        pass


###############################################################################
# Tests for Detector
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
            detection1 = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 0.1, 0.8, 0.05])
            detection2 = np.array([0.3, 0.3, 0.1, 0.1, 0.0, 0.2, 0.1, 0.3])
            predict_output = [np.array([detection1, detection2])]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)
        """
        code_used = TestDetector.test_post_process_filters_and_converts_detections.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection1 = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 0.1, 0.8, 0.05])
        detection2 = np.array([0.3, 0.3, 0.1, 0.1, 0.0, 0.2, 0.1, 0.3])
        predict_output = [np.array([detection1, detection2])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)

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
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)
        """
        code_used = TestDetector.test_post_process_detection_equal_threshold.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 0.1, 0.5, 0.05])
        predict_output = [np.array([detection])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)
        self.assertEqual(bboxes, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no bounding boxes when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertEqual(class_ids, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no class IDs when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertEqual(confidence_scores, [],
                         "\n******\nTest: test_post_process_detection_equal_threshold\nFunction: post_process()\n"
                         "Error: Expected no confidence scores when confidence equals threshold.\nCode used:\n{}\n******\n".format(code_used))
        self.assertEqual(class_scores, [],
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
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)
        """
        code_used = TestDetector.test_post_process_multiple_outputs.__doc__
        detector = Detector("storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                            "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                            self.temp_class_file.name, score_threshold=0.5)
        detector.img_width = 400
        detector.img_height = 300

        detection1 = np.array([0.4, 0.4, 0.2, 0.2, 0.0, 0.05, 0.7, 0.1])
        detection2 = np.array([0.6, 0.6, 0.1, 0.1, 0.0, 0.2, 0.6, 0.1])
        predict_output = [np.array([detection1]), np.array([detection2])]

        bboxes, class_ids, confidence_scores, class_scores = detector.post_process(predict_output, score_threshold=0.5)

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
# Tests for NMS
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
# Integration Tests for Detector and NMS using Real Images
###############################################################################
class TestDetectorIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary class file with sample class labels.
        self.temp_class_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
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

        # Define a dummy network output that is predictable.
        self.dummy_output = [np.array([[0.5, 0.5, 0.2, 0.2, 0.0, 0.6, 0.4, 0.0]])]
        self.dummy_net = MagicMock()
        self.dummy_net.getLayerNames.return_value = ["layer1", "layer2"]
        self.dummy_net.forward.return_value = self.dummy_output

    def tearDown(self):
        os.unlink(self.temp_class_file.name)

    @patch("cv2.dnn.readNet")
    def test_detector_integration_with_storage_images(self, mock_readNet):
        """
        For each JPEG image found in 'storage/test_images', run detector.predict() and
        detector.post_process() using a dummy network output, then apply NMS.filter().
        
        Also, manually call cv2.dnn.NMSBoxes on the post-processed bounding boxes and compare
        the indices with those returned by NMS.filter(). This test confirms that the calculations
        (conversion from center-based to top-left coordinates, scaling based on image dimensions,
        and NMS filtering) are accurately implemented.
        
        Code used:
            mock_readNet.return_value = self.dummy_net
            images_pattern = os.path.join("storage", "test_images", "*.jpg")
            for image_file in sorted(glob.glob(images_pattern)):
                img = cv2.imread(image_file)
                detector = Detector(..., self.temp_class_file.name, score_threshold=0.5)
                outputs = detector.predict(img)
                bboxes, class_ids, confidence_scores, class_scores = detector.post_process(outputs, score_threshold=0.5)
                nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
                filtered = nms_instance.filter(bboxes, class_ids, confidence_scores, class_scores)
        """
        code_used = TestDetectorIntegration.test_detector_integration_with_storage_images.__doc__
        mock_readNet.return_value = self.dummy_net

        images_pattern = os.path.join("storage", "test_images", "*.jpg")
        image_files = sorted(glob.glob(images_pattern))
        if not image_files:
            self.skipTest(f"No test images found in {images_pattern}")

        for image_file in image_files:
            img = cv2.imread(image_file)
            self.assertIsNotNone(img, f"\nTest: test_detector_integration_with_storage_images\nFunction: Integration\nError: Failed to load image: {image_file}\nCode used:\n{code_used}\n")
            detector = Detector(
                "storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                self.temp_class_file.name,
                score_threshold=0.5
            )
            outputs = detector.predict(img)
            H, W = img.shape[:2]
            expected_bbox = [int(0.4 * W), int(0.4 * H), int(0.2 * W), int(0.2 * H)]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(outputs, score_threshold=0.5)
            self.assertGreaterEqual(len(bboxes), 1,
                                    f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Expected at least one detection for {image_file}.\nCode used:\n{code_used}\n")
            self.assertEqual(bboxes[0], expected_bbox,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Incorrect bounding box conversion for image {image_file}. Expected {expected_bbox}, got {bboxes[0]}.\nCode used:\n{code_used}\n")
            self.assertGreaterEqual(confidence_scores[0], 0.5,
                                      f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Confidence score too low for image {image_file}.\nCode used:\n{code_used}\n")
            nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
            filtered = nms_instance.filter(bboxes, class_ids, confidence_scores, class_scores)
            indices = cv2.dnn.NMSBoxes(bboxes, confidence_scores, 0.5, 0.4)
            if len(indices) > 0:
                indices = indices.flatten().tolist()
            else:
                indices = []
            expected_filtered_bboxes = [bboxes[i] for i in indices]
            expected_filtered_class_ids = [class_ids[i] for i in indices]
            self.assertEqual(filtered[0], expected_filtered_bboxes,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: NMS.filter()\nError: Detector NMS filtered bounding boxes for {image_file} do not match manual NMS results.\nCode used:\n{code_used}\n")
            self.assertEqual(filtered[1], expected_filtered_class_ids,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: NMS.filter()\nError: Detector NMS filtered class IDs for {image_file} do not match manual NMS results.\nCode used:\n{code_used}\n")


###############################################################################
# Tests for Preprocessing (using DummyVideoCapture)
###############################################################################
class TestPreprocessing(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_capture_video_yields_every_nth_frame_from_list(self, mock_VideoCapture):
        """
        Verify that Preprocessing.capture_video yields every nth frame from a list.
        
        Code used:
            dummy_frames = [np.full((100, 100, 3), fill_value=i, dtype=np.uint8) for i in range(15)]
            drop_rate = 3
            mock_capture = DummyVideoCapture(dummy_frames)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing("dummy_path.mp4", drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())
        Example:
            With 15 dummy frames and drop_rate=3, expected frames are at indices 0, 3, 6, 9, 12.
        """
        code_used = TestPreprocessing.test_capture_video_yields_every_nth_frame_from_list.__doc__
        dummy_frames = [np.full((100, 100, 3), fill_value=i, dtype=np.uint8) for i in range(15)]
        drop_rate = 3

        mock_capture = DummyVideoCapture(dummy_frames)
        mock_VideoCapture.return_value = mock_capture

        preprocessing = Preprocessing("dummy_path.mp4", drop_rate=drop_rate)
        captured_frames = list(preprocessing.capture_video())

        expected_frames = [dummy_frames[i] for i in range(0, len(dummy_frames), drop_rate)]
        self.assertEqual(len(captured_frames), len(expected_frames),
                         f"\nTest: test_capture_video_yields_every_nth_frame_from_list\nFunction: capture_video()\nError: The number of captured frames ({len(captured_frames)}) does not match the expected count ({len(expected_frames)}).\nCode used:\n{code_used}\n")
        for cap_frame, exp_frame in zip(captured_frames, expected_frames):
            np.testing.assert_array_equal(cap_frame, exp_frame,
                                          err_msg=f"\nTest: test_capture_video_yields_every_nth_frame_from_list\nFunction: capture_video()\nError: The content of a captured frame does not match the expected frame.\nCode used:\n{code_used}\n")

    @patch('cv2.VideoCapture')
    def test_capture_video_from_directory_with_50_frames(self, mock_VideoCapture):
        """
        Verify that Preprocessing.capture_video correctly loads images from a directory
        and yields every nth frame.
        
        Code used:
            Create 50 dummy images in a temporary directory.
            drop_rate = 5
            mock_capture = DummyVideoCapture(tmpdirname)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing(tmpdirname, drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())
        Example:
            With a directory of 50 images and drop_rate=5, expected frames are at indices 0, 5, 10, ..., 45.
        """
        code_used = TestPreprocessing.test_capture_video_from_directory_with_50_frames.__doc__
        with tempfile.TemporaryDirectory() as tmpdirname:
            num_frames = 50
            frame_shape = (50, 50, 3)
            for i in range(num_frames):
                dummy_img = np.full(frame_shape, fill_value=i, dtype=np.uint8)
                filename = os.path.join(tmpdirname, f"frame_{i:03d}.png")
                cv2.imwrite(filename, dummy_img)

            drop_rate = 5

            mock_capture = DummyVideoCapture(tmpdirname)
            mock_VideoCapture.return_value = mock_capture

            preprocessing = Preprocessing(tmpdirname, drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())

            expected_frames = []
            for i in range(0, num_frames, drop_rate):
                filename = os.path.join(tmpdirname, f"frame_{i:03d}.png")
                img = cv2.imread(filename)
                if img is not None:
                    expected_frames.append(img)

            self.assertEqual(len(captured_frames), len(expected_frames),
                             f"\nTest: test_capture_video_from_directory_with_50_frames\nFunction: capture_video()\nError: The number of frames captured ({len(captured_frames)}) does not match the expected count ({len(expected_frames)}) based on the drop rate.\nCode used:\n{code_used}\n")
            for cap_frame, exp_frame in zip(captured_frames, expected_frames):
                np.testing.assert_array_equal(cap_frame, exp_frame,
                                              err_msg=f"\nTest: test_capture_video_from_directory_with_50_frames\nFunction: capture_video()\nError: The content of the captured frame does not match the expected image content.\nCode used:\n{code_used}\n")

    @patch('cv2.VideoCapture')
    def test_video_file_not_opened(self, mock_VideoCapture):
        """
        Verify that if cv2.VideoCapture cannot open a video file, capture_video raises a ValueError.
        
        Code used:
            mock_capture = DummyVideoCapture([], open_success=False)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing("nonexistent.mp4", drop_rate=5)
            list(preprocessing.capture_video())
        Example:
            For a non-existent file, the error message should mention 'Unable to open video file'.
        """
        code_used = TestPreprocessing.test_video_file_not_opened.__doc__
        mock_capture = DummyVideoCapture([], open_success=False)
        mock_VideoCapture.return_value = mock_capture

        preprocessing = Preprocessing("nonexistent.mp4", drop_rate=5)
        with self.assertRaises(ValueError, msg=f"\nTest: test_video_file_not_opened\nFunction: capture_video()\nError: If the video file cannot be opened, a ValueError must be raised.\nCode used:\n{code_used}\n"):
            list(preprocessing.capture_video())


if __name__ == '__main__':
    unittest.main()
