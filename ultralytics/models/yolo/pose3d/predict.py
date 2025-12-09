# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class Pose3dPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a Pose3d model.

    This class specializes in 3D pose estimation, handling keypoints and bone orientation detection alongside
    standard object detection capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO Pose3d model with keypoint and bone detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints and bones.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose3d import Pose3dPredictor
        >>> args = dict(model="yolo11n-pose3d.pt", source=ASSETS)
        >>> predictor = Pose3dPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize Pose3dPredictor for 3D pose estimation tasks.

        Sets up a Pose3dPredictor instance, configuring it for pose3d detection tasks and handling device-specific
        warnings for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose3d import Pose3dPredictor
            >>> args = dict(model="yolo11n-pose3d.pt", source=ASSETS)
            >>> predictor = Pose3dPredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose3d"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction, including keypoints and bones.

        Extends the parent class implementation by extracting keypoint and bone orientation data from predictions
        and adding them to the result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, keypoints, and bones with shape (N, 6+K*D+B*3)
                where N is the number of detections, K is the number of keypoints, D is the keypoint dimension,
                and B is the number of bones.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes,
                keypoints, and bone orientations.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        kpt_len = self.model.kpt_shape[0] * self.model.kpt_shape[1]
        pred_kpts = pred[:, 6:6+kpt_len].view(len(pred), *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        pred_bones = pred[:, 6+kpt_len:].view(len(pred), *self.model.bone_shape)
        result.update(keypoints=pred_kpts)
        result.update(bones=pred_bones)
        return result
