# Pose3D Augmentation Guide

## Overview
This guide explains which augmentations are safe to use with Pose3D models where bones are represented as **3D unit vectors in camera coordinate system**.

## Camera Coordinate System
- **X-axis**: Right (positive to the right of camera)
- **Y-axis**: Down (positive downward)
- **Z-axis**: Forward (positive away from camera into the scene)

## Safe Augmentations ✅

### 1. **Translation** (Image Space)
- **Effect**: Shifts the image in X/Y directions
- **Bone Impact**: None - bones are in camera space, not image space
- **Config**: `translate: 0.1` (10% of image size)

### 2. **Scaling** (Image Space)
- **Effect**: Zooms in/out on the image
- **Bone Impact**: None - bones are unit vectors (direction only)
- **Config**: `scale: 0.5` (50-150% scale range)

### 3. **Mosaic**
- **Effect**: Combines 4 images into one
- **Bone Impact**: None - each image retains its camera space
- **Config**: `mosaic: 1.0` (100% probability)

### 4. **MixUp**
- **Effect**: Blends two images together
- **Bone Impact**: None - linear interpolation preserves bone directions
- **Config**: `mixup: 0.1` (10% probability)

### 5. **CutMix**
- **Effect**: Cuts and pastes regions between images
- **Bone Impact**: None - bones stay with their respective regions
- **Config**: `cutmix: 0.1` (10% probability)

### 6. **Color Augmentations**
- **HSV**: Hue, Saturation, Value adjustments
- **Bone Impact**: None - color doesn't affect 3D geometry
- **Config**: 
  ```yaml
  hsv_h: 0.015  # Hue gain
  hsv_s: 0.7    # Saturation gain
  hsv_v: 0.4    # Value gain
  ```

### 7. **Horizontal Flip** (with bone transformation)
- **Effect**: Mirrors image left-to-right
- **Bone Impact**: **Negates X component** of bone vectors
- **Implementation**: Automatically handled in `Instances.fliplr()`
- **Config**: `fliplr: 0.5` (50% probability)
- **Requires**: `flip_idx` in dataset YAML for keypoint symmetry

## Unsafe Augmentations ⚠️

### 1. **Vertical Flip** (Complex transformation)
- **Effect**: Flips image top-to-bottom
- **Bone Impact**: Requires negating Y and Z components
- **Status**: ⚠️ **Use with caution** - may need camera-specific calibration
- **Config**: `flipud: 0.0` (disabled by default)
- **Note**: Current implementation negates Y and Z, but this assumes standard camera orientation

### 2. **Rotation** (Requires 3D rotation matrix)
- **Effect**: Rotates image by angle
- **Bone Impact**: Requires full 3D rotation matrix to transform bones
- **Status**: ❌ **NOT IMPLEMENTED** - bones won't match rotated image
- **Config**: `degrees: 0.0` (disabled)
- **Why**: Image rotation ≠ camera rotation; needs complex 3D transformation

### 3. **Shear** (Perspective distortion)
- **Effect**: Skews the image
- **Bone Impact**: Changes camera intrinsics, invalidates bone directions
- **Status**: ❌ **NOT RECOMMENDED**
- **Config**: `shear: 0.0` (disabled)

### 4. **Perspective** (Camera parameter change)
- **Effect**: Applies perspective transformation
- **Bone Impact**: Changes camera intrinsics and extrinsics
- **Status**: ❌ **NOT RECOMMENDED**
- **Config**: `perspective: 0.0` (disabled)

## Recommended Configuration

### Conservative (High Accuracy)
```yaml
# Augmentation hyperparameters
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0      # No rotation
translate: 0.1    # 10% translation
scale: 0.3        # 70-130% scale
shear: 0.0        # No shear
perspective: 0.0  # No perspective
flipud: 0.0       # No vertical flip
fliplr: 0.5       # 50% horizontal flip
mosaic: 1.0       # Always use mosaic
mixup: 0.0        # No mixup initially
cutmix: 0.0       # No cutmix initially
copy_paste: 0.0   # No copy-paste
```

### Moderate (Balanced)
```yaml
# Augmentation hyperparameters
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0      # No rotation
translate: 0.2    # 20% translation
scale: 0.5        # 50-150% scale
shear: 0.0        # No shear
perspective: 0.0  # No perspective
flipud: 0.0       # No vertical flip
fliplr: 0.5       # 50% horizontal flip
mosaic: 1.0       # Always use mosaic
mixup: 0.1        # 10% mixup
cutmix: 0.1       # 10% cutmix
copy_paste: 0.0   # No copy-paste
```

### Aggressive (Maximum Augmentation)
```yaml
# Augmentation hyperparameters
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 0.0      # No rotation
translate: 0.2    # 20% translation
scale: 0.7        # 30-170% scale
shear: 0.0        # No shear
perspective: 0.0  # No perspective
flipud: 0.0       # No vertical flip
fliplr: 0.5       # 50% horizontal flip
mosaic: 1.0       # Always use mosaic
mixup: 0.15       # 15% mixup
cutmix: 0.15      # 15% cutmix
copy_paste: 0.1   # 10% copy-paste
```

## Implementation Details

### Horizontal Flip Transformation
When `fliplr` is applied:
1. Image is mirrored left-to-right
2. Keypoint X-coordinates are flipped: `x_new = width - x_old`
3. Bone X-components are negated: `bone_x_new = -bone_x_old`
4. Keypoints are reordered according to `flip_idx` (left↔right symmetry)

### Dataset YAML Requirements
For pose3d with horizontal flip, your dataset YAML must include:
```yaml
# Required for pose3d
kpt_shape: [17, 3]  # [num_keypoints, dims]
bone_shape: [13, 3]  # [num_bones, 3D_vector_dims]

# Required for fliplr with keypoint symmetry
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# Maps: [nose, left_eye, right_eye, left_ear, right_ear, ...]
#    to: [nose, right_eye, left_eye, right_ear, left_ear, ...]
```

## Testing Augmentations

### Visual Verification
1. Train with `plots=True` to generate augmented sample images
2. Check `runs/pose3d/train/train_batch*.jpg` for visual inspection
3. Verify bones align with flipped keypoints

### Validation Metrics
Monitor these metrics to ensure augmentations help:
- **mAP50-95 (Pose)**: Should improve or stay stable
- **Bone Angular Error**: Should decrease
- **Bone Cosine Similarity**: Should increase

### Debugging
If training degrades with augmentations:
1. Disable all augmentations except color (HSV)
2. Add augmentations one at a time
3. Check if `flip_idx` is correctly defined
4. Verify bone vectors are normalized unit vectors

## Future Enhancements

### Rotation Support (Advanced)
To support rotation, you would need:
1. Extract rotation angle from augmentation
2. Create 3D rotation matrix around Z-axis (camera optical axis)
3. Apply rotation to bone vectors: `bone_new = R @ bone_old`
4. This requires modifying `RandomPerspective` class

### Vertical Flip Calibration
For vertical flip support:
1. Verify your camera's Y-axis convention (up vs down)
2. Test if current implementation (negate Y and Z) works
3. May need camera-specific calibration

## References
- Camera coordinate systems: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- YOLO augmentations: https://docs.ultralytics.com/modes/train/#augmentation-settings

