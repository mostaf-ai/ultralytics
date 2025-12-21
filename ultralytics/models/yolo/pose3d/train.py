# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models import yolo
from ultralytics.nn.tasks import Pose3dModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_results


class Pose3dTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO Pose3d models.

    This trainer specializes in handling 3D pose estimation tasks, managing model training, validation, and visualization
    of 2D pose keypoints alongside 3D bone orientations and bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (Pose3dModel): The pose3d estimation model being trained.
        data (dict): Dataset configuration including keypoint shape and bone shape information.
        loss_names (tuple): Names of the loss components used in training (box, pose, kobj, cls, dfl, bone).

    Methods:
        get_model: Retrieve a pose3d estimation model with specified configuration.
        set_model_attributes: Set keypoints and bone shape attributes on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_metrics: Generate and save training/validation metric plots.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape and bone_shape keys.

    Notes:
        The freeze parameter supports extended options for fine-grained control:
        - int: Freeze layers 0 to n-1 (e.g., freeze=10)
        - list[int]: Freeze specific layer indices (e.g., freeze=[0, 1, 2])
        - list[str]: Freeze named head components:
            'cv2' - Freeze detection box regression branch
            'cv3' - Freeze detection classification branch
            'cv4' - Freeze keypoint head
            'cv5' - Freeze bone orientation head
            'dfl' - Freeze DFL layer (already frozen by default)
            'no_dfl' - Unfreeze DFL layer (not recommended)
        - list[mixed]: Combine int and str (e.g., freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'cv2', 'cv3'])

    Examples:
        >>> from ultralytics.models.yolo.pose3d import Pose3dTrainer
        >>> args = dict(model="yolo11n-pose3d.pt", data="panoptic.yaml", epochs=100)
        >>> trainer = Pose3dTrainer(overrides=args)
        >>> trainer.train()

        # Freeze backbone (0-10) + detection head (cv2, cv3), train only cv4/cv5:
        >>> args = dict(model="yolo11n-pose3d.pt", data="panoptic.yaml", freeze=[0,1,2,3,4,5,6,7,8,9,10, "cv2", "cv3"])
        >>> trainer = Pose3dTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize a Pose3dTrainer object for training YOLO Pose3d models.

        This initializes a trainer specialized for 3D pose estimation tasks, setting the task to 'pose3d' and
        handling specific configurations needed for keypoint and bone orientation detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose3d' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.

        Examples:
            >>> from ultralytics.models.yolo.pose3d import Pose3dTrainer
            >>> args = dict(model="yolo11n-pose3d.pt", data="panoptic.yaml", epochs=100)
            >>> trainer = Pose3dTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose3d"

        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(
        self,
        cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
        weights: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Pose3dModel:
        """
        Get Pose3d estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (Pose3dModel): Initialized Pose3d estimation model with keypoint and bone prediction heads.
        """
        model = Pose3dModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], data_bone_shape=self.data["bone_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape and bone shape attributes of Pose3dModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.bone_shape = self.data["bone_shape"]

    def _setup_train(self):
        """
        Build dataloaders and optimizer on correct rank process.

        This method extends the base trainer's _setup_train to handle extended freeze parameter
        with string names like 'cv2', 'cv3', 'cv4', 'cv5'.

        The freeze parameter can be:
        - int: Freeze layers 0 to n-1 (default behavior)
        - list of int: Freeze specific layer indices
        - list of str: Freeze specific head components (cv2, cv3, cv4, cv5)
        - list of mixed: Combine int and str for flexible freezing

        Examples:
            freeze=10                              # Freeze layers 0-9
            freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Freeze backbone layers
            freeze=['cv4', 'cv5']                  # Freeze keypoint and bone heads
            freeze=[0,1,2,3,4,5,6,7,8,9,10, 'cv2', 'cv3']  # Freeze backbone + detection, train cv4/cv5

        Note:
            When string components are frozen, the optimizer is rebuilt to exclude them.
            This is necessary because the base trainer's build_optimizer() adds ALL parameters
            to the optimizer, and AdamW's decoupled weight decay would still modify frozen
            parameters during optimizer.step() even if they have requires_grad=False.
        """
        # Check if freeze contains string values that need special handling
        freeze_value = getattr(self.args, "freeze", None)
        has_string_freeze = self._has_string_freeze(freeze_value)

        if has_string_freeze:
            # Parse the freeze parameter for string components
            layer_indices, string_components = self._parse_freeze_parameter(freeze_value)

            # Temporarily set freeze to only layer indices for parent class
            original_freeze = freeze_value
            self.args.freeze = layer_indices if layer_indices else []

            # Call parent setup (handles layer index freezing)
            super()._setup_train()

            # Restore original freeze setting
            self.args.freeze = original_freeze

            # Apply string-based component freezing (cv2, cv3, cv4, cv5, etc.)
            if string_components:
                self._freeze_named_components(string_components)
                # Rebuild optimizer to exclude newly frozen parameters
                # This is critical because weight decay in AdamW is decoupled and would
                # still modify frozen parameters if they're in the optimizer
                self._rebuild_optimizer()
        else:
            # Use default behavior
            super()._setup_train()

    def _has_string_freeze(self, freeze_value) -> bool:
        """Check if freeze parameter contains any string values."""
        if freeze_value is None:
            return False
        if isinstance(freeze_value, str):
            return True
        if isinstance(freeze_value, list):
            return any(isinstance(x, str) for x in freeze_value)
        return False

    def _parse_freeze_parameter(self, freeze_value):
        """
        Parse freeze parameter into layer indices and string components.

        Args:
            freeze_value: The freeze parameter value (int, str, or list)

        Returns:
            tuple: (layer_indices: list[int], string_components: list[str])
        """
        layer_indices = []
        string_components = []

        if freeze_value is None:
            return layer_indices, string_components

        if isinstance(freeze_value, int):
            layer_indices = list(range(freeze_value))
        elif isinstance(freeze_value, str):
            string_components = [freeze_value]
        elif isinstance(freeze_value, list):
            for item in freeze_value:
                if isinstance(item, int):
                    layer_indices.append(item)
                elif isinstance(item, str):
                    string_components.append(item)

        return layer_indices, string_components

    def _freeze_named_components(self, components: list):
        """
        Freeze specific named components in the model head.

        Supported component names:
        - 'cv2': Detection box regression branch
        - 'cv3': Detection classification branch
        - 'cv4': Keypoint prediction head (from Pose)
        - 'cv5': Bone orientation head (from Pose3d)
        - 'dfl': Distribution focal loss layer (frozen by default, this is explicit)
        - 'no_dfl': Unfreeze DFL layer (NOT recommended - DFL weights are fixed math constants)

        Args:
            components (list): List of component names to freeze.

        Note:
            This method also updates `self.freeze_layer_names` so that `_model_train()` will
            properly set BatchNorm layers in frozen components to eval mode. This is critical
            because BatchNorm running statistics (running_mean, running_var) are buffers, not
            parameters, and only stop updating when the module is in eval mode.
        """
        frozen_count = 0
        unfrozen_count = 0

        # Check for 'no_dfl' - user wants to unfreeze DFL (not recommended)
        unfreeze_dfl = "no_dfl" in components
        if unfreeze_dfl:
            LOGGER.warning(
                "⚠️ 'no_dfl' specified: Unfreezing DFL layer. This is NOT recommended!\n"
                "DFL weights are fixed mathematical coefficients [0,1,2,...,15] for computing\n"
                "the expected value of box coordinate distributions. Training them may break detection."
            )
            components = [c for c in components if c != "no_dfl"]

        # Update freeze_layer_names so _model_train() will set BatchNorm layers to eval mode
        # This is critical: requires_grad=False only prevents gradient updates to learnable params,
        # but BatchNorm running_mean/running_var are buffers that update during forward pass in train mode
        freeze_patterns_to_add = []
        for component in components:
            if component != "dfl":  # dfl is already in freeze_layer_names
                freeze_patterns_to_add.append(f".{component}.")
                freeze_patterns_to_add.append(f".{component}[")

        if hasattr(self, "freeze_layer_names"):
            self.freeze_layer_names = list(self.freeze_layer_names) + freeze_patterns_to_add
        else:
            self.freeze_layer_names = freeze_patterns_to_add

        for name, param in self.model.named_parameters():
            should_freeze = False
            should_unfreeze = False

            # Handle DFL unfreezing if requested
            if unfreeze_dfl and ".dfl" in name:
                should_unfreeze = True

            for component in components:
                if component == "dfl":
                    # DFL is already frozen by default in base trainer, but allow explicit specification
                    if ".dfl" in name:
                        should_freeze = True
                elif f".{component}." in name or f".{component}[" in name or name.endswith(f".{component}"):
                    # Match component name in parameter path (e.g., .cv2., .cv4[0].)
                    should_freeze = True

            if should_unfreeze and not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
                LOGGER.info(f"⚠️ Unfreezing DFL component '{name}'")
            elif should_freeze and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
                LOGGER.info(f"Freezing component '{name}'")

        if frozen_count > 0:
            LOGGER.info(f"🔒 Frozen {frozen_count} parameters from components: {[c for c in components if c != 'no_dfl']}")
        if unfrozen_count > 0:
            LOGGER.info(f"⚠️ Unfrozen {unfrozen_count} DFL parameters (not recommended)")

    def _rebuild_optimizer(self):
        """
        Rebuild the optimizer to exclude frozen parameters.

        This is necessary because:
        1. The base trainer's build_optimizer() adds ALL parameters to the optimizer
        2. AdamW's decoupled weight decay modifies parameters during optimizer.step()
           regardless of requires_grad status
        3. If frozen parameters are in the optimizer, weight decay will still change them

        This method rebuilds the optimizer with only trainable parameters (requires_grad=True).
        """
        import math

        from torch import nn, optim

        from ultralytics.utils import LOGGER, colorstr

        model = self.model
        g = [], [], []  # optimizer parameter groups: [weight+decay, weight no decay, bias no decay]
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue  # Skip frozen parameters

                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        # Get optimizer settings from args
        name = self.args.optimizer
        lr = self.args.lr0
        momentum = self.args.momentum
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs

        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs

        if name == "auto":
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not found in list of available optimizers {optimizers}.")

        optimizer.add_param_group({"params": g[0], "weight_decay": weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

        LOGGER.info(
            f"{colorstr('optimizer:')} Rebuilt {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay}), {len(g[2])} bias(decay=0.0) "
            f"[excluded frozen parameters]"
        )

        self.optimizer = optimizer

        # Rebuild the scheduler since it sets initial_lr on param groups
        # The old scheduler was created with the old optimizer which had all parameters
        self._setup_scheduler()

    def get_validator(self):
        """Return an instance of the Pose3dValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "bone_loss"
        return yolo.pose3d.Pose3dValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot training/validation metrics."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def get_dataset(self) -> Dict[str, Any]:
        """
        Retrieve the dataset and ensure it contains the required `kpt_shape` and `bone_shape` keys.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` or `bone_shape` keys are not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose3d/")
        if "bone_shape" not in data:
            raise KeyError(f"No `bone_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose3d/")
        return data
