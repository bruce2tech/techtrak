
import unittest
import numpy as np
import inspect


from modules.utils import metrics


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

        print("Baseline:", self.baseline_map)

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
            [0, 1, 3],    # Image 1: extra detection with class 3 (false positive)
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


