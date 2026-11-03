# Running the student inside the ZED SDK (boxes)

Grip's `zed-ros2-wrapper` fork can run a YOLO-like ONNX directly in the ZED SDK
(`CUSTOM_YOLOLIKE_BOX_OBJECTS`). That gives 2D boxes plus the SDK's 3D position, velocity and
tracking per object (useful on a moving belt), but **no masks**. So this is a companion to the
mask node (`picklabel_ros`), not a replacement. Untested here: there is no ZED on this side.

## 1. Train a detection twin on the same data

```bash
# same Roboflow version / dataset as the seg student, detection head
yolo detect train data=datasets/R3/data.yaml model=yolo11s.pt imgsz=512 epochs=100 name=R3_det
```

## 2. Export in the format the SDK reads

```bash
picklabel export --weights runs/train/R3_det/weights/best.pt --kind zed --imgsz 512
# equivalent to: yolo export model=best.pt format=onnx simplify=True dynamic=False imgsz=512
```

This also writes `best.zed_custom_object_detection.yaml` with the block below.

## 3. Wrapper config

`zed_wrapper/config/common_stereo.yaml`:

```yaml
object_detection:
  od_enabled: true
  detection_model: 'CUSTOM_YOLOLIKE_BOX_OBJECTS'
  enable_tracking: true
```

`zed_wrapper/config/custom_object_detection.yaml`:

```yaml
/**:
  ros__parameters:
    object_detection:
      custom_onnx_file: '/path/to/best.onnx'
      custom_onnx_input_size: 512
      custom_class_count: 1
      class_000:
        label: 'plastic'
        model_class_id: 0
        enabled: true
        confidence_threshold: 50.0   # [0, 99]; start at 100 x the calibrated student threshold
        is_grounded: true            # lies on the belt
        is_static: false             # moves with the belt
        tracking_timeout: -1.0
        tracking_max_dist: -1.0
```

The first launch builds a TensorRT engine from the ONNX, which takes a few minutes. Detections
come out on the wrapper's `obj_det/objects` topic (`zed_msgs/ObjectsStamped`).
