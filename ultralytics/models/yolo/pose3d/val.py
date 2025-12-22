# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, Pose3dMetrics, kpt_iou


class Pose3dValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a Pose3d model.

    This validator is specifically designed for 3D pose estimation tasks, handling keypoints, bone orientations,
    and implementing specialized metrics for pose3d evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (List[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        bone_shape (List[int]): Shape of the bone orientations, typically [13, 3] for 3D unit vectors.
        args (dict): Arguments for the validator including task set to "pose3d".
        metrics (Pose3dMetrics): Metrics object for pose3d evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints and bones data to float and moving to device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose3d estimation metrics for YOLO model.
        postprocess: Postprocess YOLO predictions to extract and reshape keypoints and bones.
        _prepare_batch: Prepare a batch for processing by converting keypoints and bones to float and scaling.
        _process_batch: Return correct prediction matrix by computing IoU and adding bone metrics.
        _add_bone_metrics: Add bone orientation metrics to the metrics object.
        save_one_txt: Save YOLO pose3d detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose3d import Pose3dValidator
        >>> args = dict(model="yolo11n-pose3d.pt", data="panoptic.yaml")
        >>> validator = Pose3dValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize a Pose3dValidator object for 3D pose estimation validation.

        This validator is specifically designed for 3D pose estimation tasks, handling keypoints, bone orientations,
        and implementing specialized metrics for pose3d evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose3d".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.pose3d import Pose3dValidator
            >>> args = dict(model="yolo11n-pose3d.pt", data="panoptic.yaml")
            >>> validator = Pose3dValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with pose3d-specific functionality. It initializes with sigma values
            for OKS calculation and sets up Pose3dMetrics for evaluation. A warning is displayed when using Apple MPS
            due to a known bug with pose models.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose3d"
        self.metrics = Pose3dMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch by converting keypoints and bones data to float and moving to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        batch["bones"] = batch["bones"].to(self.device).float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10 + "%11s" * 2) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Bones(AngErr",
            "CosSim)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        self.bone_shape = self.data["bone_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Postprocess YOLO predictions to extract and reshape keypoints and bones for pose3d.

        The base detector returns a dict with 'extra' containing all task-specific outputs.
        For pose3d, 'extra' packs [kpt_flat, bone_flat]. We split and reshape both.
        """
        preds = super().postprocess(preds)
        kpt_len = self.kpt_shape[0] * self.kpt_shape[1]
        bone_len = self.bone_shape[0] * self.bone_shape[1]
        for pred in preds:
            extra = pred.pop("extra")
            if extra.numel() == 0:
                pred["keypoints"] = extra.view(0, *self.kpt_shape)
                pred["bones"] = extra.view(0, *self.bone_shape)
                continue
            pred["keypoints"] = extra[:, :kpt_len].view(-1, *self.kpt_shape)
            pred["bones"] = extra[:, kpt_len : kpt_len + bone_len].view(-1, *self.bone_shape)
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (Dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        pbatch["bones"] = batch["bones"][batch["batch_idx"] == si]
        return pbatch

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (Dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels,
                'bboxes' for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose
                true positives across 10 IoU levels.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_p = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        
        # Add bone orientation metrics
        if "bones" in preds and "bones" in batch:
            self._add_bone_metrics(preds, batch)
        
        return tp

    def _add_bone_metrics(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> None:
        """
        Add bone orientation metrics to the metrics object.

        Args:
            preds: Dictionary containing predictions with 'bones' key
            batch: Dictionary containing ground truth with 'bones' key
        """
        try:
            # Extract bone predictions and ground truth
            pred_bones = preds["bones"].cpu().numpy()
            gt_bones = batch["bones"].cpu().numpy()

            # Handle shape mismatch: pred_bones may have max_det predictions,
            # while gt_bones has actual number of ground truth instances
            n_gt = len(gt_bones)
            n_pred = len(pred_bones)

            if n_gt == 0 or n_pred == 0:
                return  # Skip if no predictions or no ground truth

            # Match predictions to ground truth using bounding box IoU
            matched_pred_indices = []
            matched_gt_indices = []
            if "bboxes" in preds and "bboxes" in batch:
                from ultralytics.utils.metrics import box_iou

                pred_bboxes = preds["bboxes"].cpu()
                gt_bboxes = batch["bboxes"].cpu()

                if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
                    iou_matrix = box_iou(pred_bboxes, gt_bboxes)

                    # Collect all potential matches above threshold
                    candidates = []
                    for gt_idx in range(n_gt):
                        for pred_idx in range(n_pred):
                            iou_val = iou_matrix[pred_idx, gt_idx].item()
                            if iou_val > 0.5:  # Only consider matches with IoU > 0.5
                                candidates.append((pred_idx, gt_idx, iou_val))

                    # Sort by IoU descending (best matches first)
                    candidates.sort(key=lambda x: x[2], reverse=True)

                    # Greedy one-to-one matching: each pred and GT used only once
                    used_preds = set()
                    used_gts = set()
                    for pred_idx, gt_idx, _ in candidates:
                        if pred_idx not in used_preds and gt_idx not in used_gts:
                            matched_pred_indices.append(pred_idx)
                            matched_gt_indices.append(gt_idx)
                            used_preds.add(pred_idx)
                            used_gts.add(gt_idx)

                    if len(matched_pred_indices) > 0:
                        pred_bones = pred_bones[matched_pred_indices]
                        gt_bones = gt_bones[matched_gt_indices]
                    else:
                        return  # No valid matches
                else:
                    return
            else:
                # Fallback: truncate to minimum length
                min_len = min(n_pred, n_gt)
                pred_bones = pred_bones[:min_len]
                gt_bones = gt_bones[:min_len]

            # Validate bone shape consistency
            if pred_bones.shape[1:] != gt_bones.shape[1:]:
                LOGGER.warning(
                    f"Bone shape mismatch: pred {pred_bones.shape} vs gt {gt_bones.shape}. Skipping bone metrics."
                )
                return

            # Add bone metrics to the metrics object
            self.metrics.add_bone_metrics(pred_bones, gt_bones)

        except Exception as e:
            LOGGER.warning(f"Failed to add bone metrics: {e}")

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', 'cls' and 'keypoints.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format
        to COCO format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (Dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls',
                and 'keypoints' tensors.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = ops.scale_coords(
            pbatch["imgsz"],
            predn["keypoints"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate object detection model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
