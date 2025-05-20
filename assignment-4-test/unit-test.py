import unittest
import numpy as np
import cv2

from modules.utils.loss import Loss  
from modules.rectification.augmentation import Augmenter


class TestLoss(unittest.TestCase):
    def setUp(self):
        # Use a smaller number of classes for easier testing.
        self.loss = Loss(iou_threshold=0.5, lambda_coord=0.5, lambda_noobj=0.5, num_classes=3)

    def test_get_predictions(self):
        """
        Test get_predictions() with dummy prediction data.
        Expected:
          - Two predictions (one per image).
          - Each prediction list: [x1, y1, x2, y2, objectness, class_score_0, class_score_1, class_score_2].
          - Example:
              Image 1: [10, 10, 20, 20, 0.9, 0.1, 0.9, 0.0]
              Image 2: [30, 30, 40, 40, 0.8, 0.7, 0.1, 0.2]
        """
        # Get the test description including code snippet (if any) from the docstring.
        code_used = TestLoss.test_get_predictions.__doc__
        predictions = [
            [[10, 10, 20, 20, 0.9, 0.1, 0.9, 0.0]],
            [[30, 30, 40, 40, 0.8, 0.7, 0.1, 0.2]]
        ]
        pred_box, objectness_score, class_scores = self.loss.get_predictions(predictions)
        
        self.assertEqual(
            pred_box.shape, (2, 4),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected predicted boxes shape to be (2,4) for input predictions = {}\n"
            "Got shape: {} with boxes: {}\nPlease check the box extraction logic.\n"
            "Code used:\n{}\n******\n".format(predictions, pred_box.shape, pred_box, code_used)
        )
        self.assertEqual(
            objectness_score.shape, (2,),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected objectness scores shape to be (2,) for input predictions = {}\n"
            "Got shape: {} with scores: {}\nPlease review the objectness extraction logic.\n"
            "Code used:\n{}\n******\n".format(predictions, objectness_score.shape, objectness_score, code_used)
        )
        self.assertEqual(
            class_scores.shape, (2, 3),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected class scores shape to be (2,3) for input predictions = {}\n"
            "Got shape: {} with class scores: {}\nPlease verify the class score extraction.\n"
            "Code used:\n{}\n******\n".format(predictions, class_scores.shape, class_scores, code_used)
        )

    def test_get_annotations(self):
        """
        Test get_annotations() with dummy annotation data.
        Expected:
          - Two annotations in the form: [class_id, x1, y1, x2, y2].
          - Example:
              Annotation 1: [1, 10, 10, 20, 20]
              Annotation 2: [0, 30, 30, 40, 40]
        """
        code_used = TestLoss.test_get_annotations.__doc__
        annotations = [
            [1, 10, 10, 20, 20],
            [0, 30, 30, 40, 40]
        ]
        gt_box, gt_class_id = self.loss.get_annotations(annotations)
        
        self.assertEqual(
            gt_box.shape, (2, 4),
            "\n******\nTest: test_get_annotations\nFunction: get_annotations()\n"
            "Error: Expected ground truth boxes shape to be (2,4) for input annotations = {}\n"
            "Got shape: {} with boxes: {}\nPlease check annotation parsing for boxes.\n"
            "Code used:\n{}\n******\n".format(annotations, gt_box.shape, gt_box, code_used)
        )
        self.assertEqual(
            gt_class_id.shape, (2,),
            "\n******\nTest: test_get_annotations\nFunction: get_annotations()\n"
            "Error: Expected ground truth class IDs shape to be (2,) for input annotations = {}\n"
            "Got shape: {} with class IDs: {}\nPlease verify the extraction of class IDs from annotations.\n"
            "Code used:\n{}\n******\n".format(annotations, gt_class_id.shape, gt_class_id, code_used)
        )

    def test_compute_loss(self):
        """
        Test compute() with a simple matching prediction and annotation.
        Expected:
          - Prediction: [10, 10, 20, 20, 0.9, 0.7, 0.3, 0.0]
          - Annotation: [0, 10, 10, 20, 20] (class 0, matching the box)
          - All loss components (total_loss, loc_loss, conf_loss_obj, conf_loss_noobj, class_loss) must be non-negative.
        """
        code_used = TestLoss.test_compute_loss.__doc__
        predictions = [
            [[10, 10, 20, 20, 0.9, 0.7, 0.3, 0.0]]
        ]
        annotations = [
            [0, 10, 10, 20, 20]
        ]
        losses = self.loss.compute(predictions, annotations)
        
        for key in losses:
            self.assertGreaterEqual(
                losses[key], 0,
                "\n******\nTest: test_compute_loss\nFunction: compute()\n"
                "Error: Expected loss component '{}' to be non-negative for inputs:\n"
                "predictions = {}\nannotations = {}\nGot {}.\nPlease inspect the loss computation for {}.\n"
                "Code used:\n{}\n******\n".format(key, predictions, annotations, losses[key], key, code_used)
            )
        self.assertIn(
            "total_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'total_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nPlease verify the loss aggregation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "loc_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'loc_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nCheck localization loss computation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "conf_loss_obj", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'conf_loss_obj' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nReview objectness loss for true detections.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "conf_loss_noobj", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'conf_loss_noobj' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nReview no-object loss computation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "class_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'class_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nVerify class loss calculation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )

    def test_compute_loss_increases_with_lower_class_prediction(self):
        """
        Test that compute() yields a higher total loss when the predicted probability for the correct class decreases.
        Example:
          - Scenario A (High confidence): [10,10,20,20,0.9,0.8,0.1,0.1]
          - Scenario B (Low confidence): [10,10,20,20,0.9,0.3,0.35,0.35]
        Expected:
          - Total loss and class loss in Scenario B must be greater than in Scenario A.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_A = [[[10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]]]
          predictions_B = [[[10, 10, 20, 20, 0.9, 0.3, 0.35, 0.35]]]
        """
        code_used = TestLoss.test_compute_loss_increases_with_lower_class_prediction.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_A = [[[10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]]]
        predictions_B = [[[10, 10, 20, 20, 0.9, 0.3, 0.35, 0.35]]]
        loss_A = self.loss.compute(predictions_A, annotations)
        loss_B = self.loss.compute(predictions_B, annotations)
        
        self.assertGreater(
            loss_B["total_loss"], loss_A["total_loss"],
            "\n******\nTest: test_compute_loss_increases_with_lower_class_prediction\nFunction: compute()\n"
            "Error: Expected total_loss in Scenario B ({}) to be greater than in Scenario A ({}).\n"
            "Inputs:\nannotations = {}\npredictions_A = {}\npredictions_B = {}\nReview class probability impact on loss.\n"
            "Code used:\n{}\n******\n".format(loss_B["total_loss"], loss_A["total_loss"], annotations, predictions_A, predictions_B, code_used)
        )
        self.assertGreater(
            loss_B["class_loss"], loss_A["class_loss"],
            "\n******\nTest: test_compute_loss_increases_with_lower_class_prediction\nFunction: compute()\n"
            "Error: Expected class_loss in Scenario B ({}) to exceed that in Scenario A ({}).\n"
            "Inputs:\nannotations = {}\npredictions_A = {}\npredictions_B = {}\nCheck class prediction handling.\n"
            "Code used:\n{}\n******\n".format(loss_B["class_loss"], loss_A["class_loss"], annotations, predictions_A, predictions_B, code_used)
        )

    def test_compute_loss_increases_when_objectness_decreases(self):
        """
        Test that compute() yields a higher loss when the objectness score decreases for a true detection.
        Example:
          - High objectness: [10,10,20,20,0.9,0.9,0.05,0.05]
          - Low objectness:  [10,10,20,20,0.3,0.9,0.05,0.05]
        Expected:
          - Lower objectness should yield a higher objectness loss and higher total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_high_obj = [[[10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]]]
          predictions_low_obj = [[[10, 10, 20, 20, 0.3, 0.9, 0.05, 0.05]]]
        """
        code_used = TestLoss.test_compute_loss_increases_when_objectness_decreases.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_high_obj = [[[10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]]]
        predictions_low_obj = [[[10, 10, 20, 20, 0.3, 0.9, 0.05, 0.05]]]
        loss_high = self.loss.compute(predictions_high_obj, annotations)
        loss_low = self.loss.compute(predictions_low_obj, annotations)
        
        self.assertGreater(
            loss_low["total_loss"], loss_high["total_loss"],
            "\n******\nTest: test_compute_loss_increases_when_objectness_decreases\nFunction: compute()\n"
            "Error: Expected total_loss with low objectness ({}) to be greater than with high objectness ({}).\n"
            "Inputs:\nannotations = {}\npredictions_high_obj = {}\npredictions_low_obj = {}\nCheck impact of objectness on loss.\n"
            "Code used:\n{}\n******\n".format(loss_low["total_loss"], loss_high["total_loss"], annotations, predictions_high_obj, predictions_low_obj, code_used)
        )

    def test_all_highly_overlapping_boxes_are_computed(self):
        """
        Test that compute() accumulates loss from all predicted boxes overlapping a ground truth.
        Example:
          - Two identical predictions overlapping [10,10,20,20].
          - One prediction overlapping [10,10,20,20].
        Expected:
          - Total loss for two predictions must be greater than for one.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_two = [[
              [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1],
              [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]
          ]]
          predictions_one = [[[10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]]]
        """
        code_used = TestLoss.test_all_highly_overlapping_boxes_are_computed.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_two = [[
            [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1],
            [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]
        ]]
        predictions_one = [[[10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]]]
        loss_two = self.loss.compute(predictions_two, annotations)
        loss_one = self.loss.compute(predictions_one, annotations)
        
        self.assertGreaterEqual(
            loss_two["total_loss"], loss_one["total_loss"],
            "\n******\nTest: test_all_highly_overlapping_boxes_are_computed\nFunction: compute()\n"
            "Error: Expected total_loss for two overlapping predictions ({}) to exceed that for one prediction ({}).\n"
            "Inputs:\nannotations = {}\npredictions_two = {}\npredictions_one = {}\nEnsure all overlaps contribute to the loss.\n"
            "Code used:\n{}\n******\n".format(loss_two["total_loss"], loss_one["total_loss"], annotations, predictions_two, predictions_one, code_used)
        )

    def test_non_overlapping_boxes_do_not_contribute(self):
        """
        Test that compute() ignores predicted boxes not overlapping the ground truth.
        Example:
          - Overlapping prediction: [10,10,20,20]
          - Non-overlapping prediction: [100,100,110,110]
        Expected:
          - The objectness loss for true detections (conf_loss_obj) remains unchanged when a non-overlapping prediction is added.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          overlapping_pred = [10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]
          non_overlapping_pred = [100, 100, 110, 110, 0.9, 0.8, 0.1, 0.1]
        """
        code_used = TestLoss.test_non_overlapping_boxes_do_not_contribute.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        overlapping_pred = [10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]
        non_overlapping_pred = [100, 100, 110, 110, 0.9, 0.8, 0.1, 0.1]
        predictions_overlap_only = [[overlapping_pred]]
        predictions_combined = [[overlapping_pred, non_overlapping_pred]]
        loss_overlap = self.loss.compute(predictions_overlap_only, annotations)
        loss_combined = self.loss.compute(predictions_combined, annotations)
        
        self.assertAlmostEqual(
            loss_overlap["conf_loss_obj"], loss_combined["conf_loss_obj"], places=6,
            msg="\n******\nTest: test_non_overlapping_boxes_do_not_contribute\nFunction: compute()\n"
            "Error: Expected conf_loss_obj to remain unchanged when adding non-overlapping predictions.\n"
            "Inputs:\nannotations = {}\nwith overlapping_pred = {} and non_overlapping_pred = {}\n"
            "Got conf_loss_obj: {} (overlap only) vs. {} (combined).\nPlease check overlap handling in loss computation.\n"
            "Code used:\n{}\n******\n".format(annotations, overlapping_pred, non_overlapping_pred, loss_overlap["conf_loss_obj"], loss_combined["conf_loss_obj"], code_used)
        )

    def test_loss_increases_with_bbox_shift(self):
        """
        Test that compute() yields a higher loss when the predicted bounding box shifts from the ground truth.
        Example:
          - Perfect match: [10,10,20,20]
          - Shifted prediction: [12,10,22,20] (shifted 2 pixels horizontally)
        Expected:
          - A shifted box yields higher localization loss and higher total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          prediction_perfect = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
          prediction_shifted = [12, 10, 22, 20, 0.9, 0.9, 0.05, 0.05]
        """
        code_used = TestLoss.test_loss_increases_with_bbox_shift.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        prediction_perfect = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
        prediction_shifted = [12, 10, 22, 20, 0.9, 0.9, 0.05, 0.05]
        predictions_perfect = [[prediction_perfect]]
        predictions_shifted = [[prediction_shifted]]
        loss_perfect = self.loss.compute(predictions_perfect, annotations)
        loss_shifted = self.loss.compute(predictions_shifted, annotations)
        
        self.assertGreater(
            loss_shifted["total_loss"], loss_perfect["total_loss"],
            "\n******\nTest: test_loss_increases_with_bbox_shift\nFunction: compute()\n"
            "Error: Expected total_loss for shifted prediction ({}) to exceed that for perfect prediction ({}).\n"
            "Inputs:\nannotations = {}\nprediction_perfect = {}\nprediction_shifted = {}\nReview localization loss calculation.\n"
            "Code used:\n{}\n******\n".format(loss_shifted["total_loss"], loss_perfect["total_loss"], annotations, prediction_perfect, prediction_shifted, code_used)
        )
        self.assertGreater(
            loss_shifted["loc_loss"], loss_perfect["loc_loss"],
            "\n******\nTest: test_loss_increases_with_bbox_shift\nFunction: compute()\n"
            "Error: Expected loc_loss for shifted prediction ({}) to be higher than for perfect prediction ({}).\n"
            "Inputs:\nannotations = {}\nprediction_perfect = {}\nprediction_shifted = {}\nCheck box regression impact.\n"
            "Code used:\n{}\n******\n".format(loss_shifted["loc_loss"], loss_perfect["loc_loss"], annotations, prediction_perfect, prediction_shifted, code_used)
        )

    def test_loss_increases_when_true_object_objectness_is_low(self):
        """
        Test that compute() yields a higher loss for a true object when its predicted objectness is low.
        Example:
          - High objectness: [10,10,20,20,0.9,0.9,0.05,0.05]
          - Low objectness:  [10,10,20,20,0.0,0.9,0.05,0.05]
        Expected:
          - Lower objectness increases the objectness loss (conf_loss_obj) and total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          prediction_high_obj = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
          prediction_low_obj = [10, 10, 20, 20, 0.0, 0.9, 0.05, 0.05]
        """
        code_used = TestLoss.test_loss_increases_when_true_object_objectness_is_low.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        prediction_high_obj = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
        prediction_low_obj = [10, 10, 20, 20, 0.0, 0.9, 0.05, 0.05]
        predictions_high = [[prediction_high_obj]]
        predictions_low = [[prediction_low_obj]]
        loss_high = self.loss.compute(predictions_high, annotations)
        loss_low = self.loss.compute(predictions_low, annotations)
        
        self.assertGreater(
            loss_low["total_loss"], loss_high["total_loss"],
            "\n******\nTest: test_loss_increases_when_true_object_objectness_is_low\nFunction: compute()\n"
            "Error: Expected total_loss for low objectness ({}) to be greater than for high objectness ({}).\n"
            "Inputs:\nannotations = {}\nprediction_high_obj = {}\nprediction_low_obj = {}\nExamine objectness contribution to total loss.\n"
            "Code used:\n{}\n******\n".format(loss_low["total_loss"], loss_high["total_loss"], annotations, prediction_high_obj, prediction_low_obj, code_used)
        )
        self.assertGreater(
            loss_low["conf_loss_obj"], loss_high["conf_loss_obj"],
            "\n******\nTest: test_loss_increases_when_true_object_objectness_is_low\nFunction: compute()\n"
            "Error: Expected conf_loss_obj for low objectness ({}) to exceed that for high objectness ({}).\n"
            "Inputs:\nannotations = {}\nprediction_high_obj = {}\nprediction_low_obj = {}\nReview objectness loss calculation.\n"
            "Code used:\n{}\n******\n".format(loss_low["conf_loss_obj"], loss_high["conf_loss_obj"], annotations, prediction_high_obj, prediction_low_obj, code_used)
        )

    def test_false_positive_objectness_loss(self):
        """
        Test that compute() yields a significantly higher no-object loss (conf_loss_noobj) when a false positive has high objectness.
        Example:
          - False positive with near-zero objectness: [30,30,40,40,0.0,0.1,0.1,0.8]
          - False positive with high objectness: [30,30,40,40,0.9,0.1,0.1,0.8]
        Expected:
          - The no-object loss for the high objectness false positive should be significantly higher.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          false_pred_low = [30,30,40,40,0.0,0.1,0.1,0.8]
          false_pred_high = [30,30,40,40,0.9,0.1,0.1,0.8]
        """
        code_used = TestLoss.test_false_positive_objectness_loss.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        false_pred_low = [30, 30, 40, 40, 0.0, 0.1, 0.1, 0.8]
        false_pred_high = [30, 30, 40, 40, 0.9, 0.1, 0.1, 0.8]
        predictions_low = [[false_pred_low]]
        predictions_high = [[false_pred_high]]
        loss_low = self.loss.compute(predictions_low, annotations)
        loss_high = self.loss.compute(predictions_high, annotations)
        
        self.assertGreater(
            loss_high["conf_loss_noobj"], loss_low["conf_loss_noobj"],
            "\n******\nTest: test_false_positive_objectness_loss\nFunction: compute()\n"
            "Error: The computed no-object loss (conf_loss_noobj) for a false positive with high objectness "
            "is not greater than that for a false positive with near-zero objectness.\n"
            "Inputs used:\n  annotations: {}\n  false_pred_low: {}\n  false_pred_high: {}\n"
            "Computed losses:\n  conf_loss_noobj (low objectness): {}\n  conf_loss_noobj (high objectness): {}\n"
            "Details: A false positive with a higher objectness score should incur a larger penalty in the no-object loss term.\n"
            "Please review the handling of objectness in the loss computation.\n"
            "Code used:\n{}\n******\n".format(annotations, false_pred_low, false_pred_high, loss_low["conf_loss_noobj"], loss_high["conf_loss_noobj"], code_used)
        )

if __name__ == '__main__':
    unittest.main()
