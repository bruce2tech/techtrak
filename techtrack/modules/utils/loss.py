import numpy as np
from itertools import chain
from .metrics import calculate_iou


class Loss:
    """
    *Modified* YOLO Loss for Hard Negative Mining.

    Attributes:
        num_classes (int): Number of classes.
        iou_threshold (float): Intersection over Union (IoU) threshold.
        lambda_coord (float): Weighting factor for localization loss.
        lambda_noobj (float): Weighting factor for no object confidence loss.
    """

    def __init__(self, iou_threshold=0.5, lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.5, lambda_cls=0.5, num_classes=20):
        """
        Initialize the Loss object with the given parameters.

        Internal Process:
        1. Stores the provided hyperparameters as instance attributes.
        2. Defines the column names for loss components to track them in results.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weighting factor for localization loss.
            lambda_obj (float): Weighting factor for objectness loss.
            lambda_noobj (float): Weighting factor for no object confidence loss.
            lambda_cls (float): Weighting factor for classification loss.
        """

        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.columns = [
            'total_loss', 
            'loc_loss', 
            'conf_loss_obj', 
            'conf_loss_noobj', 
            'class_loss'
        ]
        self.iou_threshold = iou_threshold
    
    def get_predictions(self, predictions):
        """
        Extracts bounding box coordinates, objectness scores, and class scores from predictions.

        Internal Process:
        1. Iterates over predictions to extract bounding box coordinates.
        2. Extracts objectness scores.
        3. Extracts class scores.

        Args:
            predictions (list): List of predicted bounding boxes and associated scores.
        
        Returns:
            tuple: (bounding boxes, objectness scores, class scores)
        """
        """
        Split raw predictions into bboxes, objectness scores, and class scores.
        """
        flat = list(chain.from_iterable(predictions))
        if not flat:
            return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,self.num_classes))
        preds = np.array(flat, float)
        bboxes = preds[:, 0:4]            # [x1,y1,x2,y2]
        obj    = preds[:, 4]
        cls    = preds[:, 5:5+self.num_classes]
        return bboxes, obj, cls
    
    def get_annotations(self, annotations):
        """
        Extract ground truth bounding boxes and class IDs from annotations.
        
        Internal Process:
        1. Iterates over annotations to extract bounding box coordinates.
        2. Extracts the corresponding class labels.
        
        Args:
            annotations (list): List of ground truth annotations.
        
        Returns:
            tuple: (ground truth bounding boxes, class labels)
        """
        gt_bboxes = []
        gt_classes = []
        for ann in annotations:
            if isinstance(ann, (list, tuple, np.ndarray)) and len(ann) == 5:
                # detect format by checking whether index 0 is integer class
                if isinstance(ann[0], (int, np.integer)):
                    # [class_id, x1, y1, x2, y2]
                    cls  = int(ann[0])
                    bbox = ann[1:5]
                else:
                    # [cx, cy, w, h, class_id]
                    bbox = ann[0:4]
                    cls  = int(ann[4])
            else:
                # assume (bbox, cls)
                bbox, cls = ann
            gt_bboxes.append(bbox)
            gt_classes.append(cls)
        return np.array(gt_bboxes, float), np.array(gt_classes, int)

    def compute(self, predictions, annotations):
        """
        Compute the YOLO loss components.

        Internal Process:
        1. Extracts predictions and annotations of a single image/frame.
        2. Iterates through annotations to compute localization, confidence, and class loss.
        3. Computes total loss using predefined weighting factors.

        Args:
            predictions (list): List of predictions of a single image.
            annotations (list): List of ground truth annotations of a single image.

        Returns:
            dict: Dictionary containing the computed loss components.
        """
        loc_loss = 0 # localization loss
        class_loss = 0 # classification loss
        conf_loss_obj = 0 # with object (or confidence) loss
        conf_loss_noobj = 0 # no object (or confidence) loss
        total_loss = 0 # aggregate loss including loc_loss, class_loss, conf_loss_obj, etc.

        # TASK: Complete this method to compute the Loss function.
        #         This method calculates the localization, objectness 
        #         (or confidence) and classification loss.
        #         This method will be called in the HardNegativeMiner class.
        #         ----------------------------------------------------------
        #         HINT: For simplicity complete use get_predictions(), get_annotations().
        #         You may add class methods to improve the readability of this code. 

        # extract
        pred_bboxes, pred_obj, pred_cls = self.get_predictions(predictions)
        gt_bboxes, gt_classes           = self.get_annotations(annotations)

        # Build IoU matrix
        if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
            iou_mat = np.zeros((len(pred_bboxes), len(gt_bboxes)), float)
            for i, pb in enumerate(pred_bboxes):
                for j, gb in enumerate(gt_bboxes):
                    iou_mat[i, j] = calculate_iou(pb, gb)
        else:
            iou_mat = np.zeros((len(pred_bboxes), len(gt_bboxes)), float)

        matched_gt = set()
        # For each prediction decide TP vs FP
        for i in range(len(pred_bboxes)):
            # find best GT match
            if gt_bboxes.shape[0] > 0:
                best_j = int(np.argmax(iou_mat[i]))
                best_iou = iou_mat[i, best_j]
            else:
                best_j, best_iou = -1, 0.0

            # check if it’s an object prediction
            if best_iou >= self.iou_threshold and best_j not in matched_gt:
                # we have localized a ground‐truth object (regardless of predicted class)
                matched_gt.add(best_j)

                # 1) Localization loss (MSE on bbox coords)
                loc_loss += np.sum((pred_bboxes[i] - gt_bboxes[best_j])**2)

                # 2) Objectness loss → label is 1 here
                conf_loss_obj += (1.0 - pred_obj[i])**2

                # 3) Always compute classification loss vs the true class
                one_hot = np.zeros(self.num_classes)
                one_hot[gt_classes[best_j]] = 1.0
                class_loss += np.sum((pred_cls[i] - one_hot)**2)

            else:
                # either low IoU or this GT already matched: treat as background
                conf_loss_noobj += (pred_obj[i])**2



        # Weighted sum
        total_loss = (
            self.lambda_coord * loc_loss +
            self.lambda_obj   * conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            self.lambda_cls   * class_loss
        )

        return {
            "total_loss": total_loss, 
            "loc_loss": loc_loss, 
            "conf_loss_obj": conf_loss_obj, 
            "conf_loss_noobj": conf_loss_noobj, 
            "class_loss": class_loss
        }
