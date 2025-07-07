import numpy as np
import cv2
from typing import List, Tuple


class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.

        :ivar self.score_threshold: The threshold below which detections are discarded.
        :ivar self.nms_iou_threshold: The IoU threshold that determines whether two boxes 
                                      are considered redundant.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def _iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between a box and a list of boxes.
        box:    [x1, y1, x2, y2]
        boxes:  [[x1, y1, x2, y2], ...]
        returns: IoU array of shape (len(boxes),)
        """
        
        # Intersection
        ix1 = np.maximum(box[0], boxes[:, 0])
        iy1 = np.maximum(box[1], boxes[:, 1])
        ix2 = np.minimum(box[2], boxes[:, 2])
        iy2 = np.minimum(box[3], boxes[:, 3])

        iw = np.maximum(ix2 - ix1 + 1, 0)
        ih = np.maximum(iy2 - iy1 + 1, 0)
        inter = iw * ih

        # Union
        area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        union = area_box + area_boxes - inter

        return inter / union
    
    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[float],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.

        :return: A tuple containing:
            - **filtered_bboxes (List[List[int]])**: The final bounding boxes after NMS.
            - **filtered_class_ids (List[int])**: The class IDs of retained bounding boxes.
            - **filtered_scores (List[float])**: The confidence scores of retained bounding boxes.
            - **filtered_class_scores (List[float])**: The class-specific scores of retained boxes.

        **How NMS Works:**
        - The function selects the bounding box with the highest confidence.
        - It suppresses any boxes that have a high IoU (overlapping area) with this selected box.
        - This process is repeated until all valid boxes are retained.

        **Example Usage:**
        ```python
        nms_processor = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        final_bboxes, final_class_ids, final_scores, final_class_scores = nms_processor.filter(
            bboxes, class_ids, scores, class_scores
        )
        ```
        """


               # 1) Try to use OpenCV's NMSBoxes if present
        if hasattr(cv2.dnn, "NMSBoxes"):
            idxs = cv2.dnn.NMSBoxes(
                bboxes, scores,
                self.score_threshold,
                self.nms_iou_threshold
            )
            # Normalize the output into a flat list of ints
            if isinstance(idxs, (list, tuple, np.ndarray)) and len(idxs) > 0:
                flat = []
                # Some versions return [[i],[j]], some return [i,j]
                for elt in idxs:
                    if isinstance(elt, (list, tuple, np.ndarray)):
                        flat.append(int(elt[0]))
                    else:
                        flat.append(int(elt))
                # Map them directly and return
                return (
                    [bboxes[i]      for i in flat],
                    [class_ids[i]   for i in flat],
                    [scores[i]      for i in flat],
                    [class_scores[i]for i in flat]
                )
            else:
                # No indices → empty
                return [], [], [], []

        # 2) Fallback: pure-NumPy NMS
        if not bboxes:
            return [], [], [], []

        orig_bboxes       = list(bboxes)
        orig_class_ids    = list(class_ids)
        orig_scores       = list(scores)
        orig_class_scores = list(class_scores)

        # Convert to [x1, y1, x2, y2]
        arr = np.array(orig_bboxes, dtype=float)
        x1 = arr[:, 0]
        y1 = arr[:, 1]
        x2 = arr[:, 0] + arr[:, 2]
        y2 = arr[:, 1] + arr[:, 3]
        dets = np.stack([x1, y1, x2, y2], axis=1)

        scr_arr = np.array(orig_scores, dtype=float)
        keep = scr_arr >= self.score_threshold
        dets = dets[keep]
        scr_arr = scr_arr[keep]
        indices = np.nonzero(keep)[0]

        order = scr_arr.argsort()[::-1]
        dets = dets[order]
        indices = indices[order]

        keep_inds = []
        while dets.shape[0] > 0:
            idx = indices[0]
            keep_inds.append(idx)
            if dets.shape[0] == 1:
                break
            ious = self._iou(dets[0], dets[1:])
            below = np.where(ious < self.nms_iou_threshold)[0]
            dets = dets[below + 1]
            indices = indices[below + 1]

        keep_inds.sort()
        return (
            [orig_bboxes[i]       for i in keep_inds],
            [orig_class_ids[i]    for i in keep_inds],
            [orig_scores[i]       for i in keep_inds],
            [orig_class_scores[i] for i in keep_inds]
        )