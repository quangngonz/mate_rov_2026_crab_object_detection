# Crab Object Detection System

A complete pipeline for detecting multiple crabs in images, trained from only three single-crab reference images using synthetic data generation.

## 🎯 Project Overview

This system demonstrates how to build a robust object detector starting from minimal data:

- **Input**: 3 reference images (1 crab each)
- **Process**: Generate synthetic training data with extensive augmentation
- **Output**: YOLOv8 model capable of detecting multiple crabs in complex scenes

## 📁 Directory Structure

```
crab_classifier/
├── reference_images/          # Original reference images (3 crabs)
│   ├── European_Green_Crab_Image.png
│   ├── Jonah_crab.png
│   └── Native_Rock_Crab.png
├── test_images/               # Test images for evaluation
│   ├── test_0.png
│   ├── test_1.png
│   ├── test_2.png
│   └── test_3.png
├── dataset/                   # Generated synthetic dataset (created by script)
│   ├── images/
│   │   ├── train/            # 1000 training images
│   │   └── val/              # 200 validation images
│   ├── labels/
│   │   ├── train/            # YOLO format annotations
│   │   └── val/
│   └── data.yaml             # Dataset configuration
├── runs/                      # Training outputs (created during training)
│   └── detect/
│       └── crab_detector/
│           ├── weights/
│           │   ├── best.pt   # Best model checkpoint
│           │   └── last.pt   # Last epoch checkpoint
│           └── results.png   # Training curves
├── detections/                # Inference results (created during inference)
│   └── detected_*.png        # Test images with bounding boxes
├── generate_dataset.py        # Synthetic data generation script
├── train.py                   # Model training script
├── inference.py               # Detection inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python generate_dataset.py
```

**What this does:**

- Segments crabs from reference images using AI-powered background removal
- Applies extensive augmentations:
  - Rotation: 0-360° (full rotation)
  - Scale: 0.3-1.5× (size variation)
  - Flip: Horizontal & vertical (50% each)
  - Color jitter: Hue, saturation, brightness, contrast
  - Noise: Gaussian noise
  - Blur: Gaussian blur
- Generates 1000 training + 200 validation images
- Creates random backgrounds and composes multiple crabs per image
- Outputs YOLO format annotations (normalized bounding boxes)

**Expected output:**

```
dataset/
├── images/train/  (1000 images)
├── images/val/    (200 images)
├── labels/train/  (1000 .txt files)
├── labels/val/    (200 .txt files)
└── data.yaml
```

### 3. Train the Model

```bash
python train.py
```

**Training configuration:**

- **Model**: YOLOv8 Nano (lightweight, fast)
- **Image size**: 640×640
- **Batch size**: 16 (adjust based on GPU memory)
- **Epochs**: 100 (with early stopping patience=50)
- **Pretrained**: Yes (COCO weights for transfer learning)
- **Device**: Auto-detect GPU/CPU

**Why YOLOv8n?**

1. Lightweight & fast (suitable for edge devices like ROVs)
2. Pre-trained on COCO (strong feature extractors)
3. Excellent for small datasets via transfer learning
4. Built-in augmentation (mosaic, mixup, HSV)
5. Native multi-object detection capability

**Expected duration:**

- GPU: ~20-30 minutes
- CPU: 2-4 hours

**Output:**

```
runs/detect/crab_detector/
├── weights/
│   ├── best.pt    # Use this for inference
│   └── last.pt
├── results.png    # Training curves
└── confusion_matrix.png
```

### 4. Run Inference

```bash
python inference.py
```

**What this does:**

- Loads the best trained model
- Runs detection on all images in `test_images/`
- Draws bounding boxes with confidence scores
- Saves annotated images to `detections/`

**Expected output:**

```
detections/
├── detected_test_0.png
├── detected_test_1.png
├── detected_test_2.png
└── detected_test_3.png
```

## 📊 Data Augmentation Strategy

### Rationale

Starting with only 3 reference images, we need extensive augmentation to:

1. Increase dataset size (avoid overfitting)
2. Simulate real-world variations (scale, rotation, lighting)
3. Handle domain gap between synthetic and real images

### Augmentation Pipeline

| Augmentation        | Range/Probability | Rationale                                      |
| ------------------- | ----------------- | ---------------------------------------------- |
| **Rotation**        | 0-360°            | Crabs can appear at any orientation underwater |
| **Scale**           | 0.3-1.5×          | Crabs appear at different distances            |
| **Horizontal Flip** | 50%               | Left-right symmetry                            |
| **Vertical Flip**   | 50%               | Crabs can be upside down                       |
| **Hue Shift**       | ±30°              | Lighting variations underwater                 |
| **Saturation**      | ±40%              | Water clarity affects color saturation         |
| **Brightness**      | ±30%              | Depth and lighting conditions                  |
| **Contrast**        | ±30%              | Camera settings and conditions                 |
| **Gaussian Noise**  | σ=0-15            | Sensor noise                                   |
| **Gaussian Blur**   | kernel 0-5        | Motion blur, focus issues                      |
| **Background**      | Random            | Varied underwater environments                 |
| **Multi-object**    | 1-8 crabs         | Realistic crowded scenes                       |

### Synthetic Data Generation Process

1. **Segmentation**: Remove background using `rembg` (U2-Net)
2. **Crop**: Extract tight bounding box around crab
3. **Augment**: Apply random transformations
4. **Compose**: Place 1-8 crabs on random background
5. **Annotate**: Generate YOLO format labels (class_id x_center y_center width height)

## 🎓 Model Architecture & Training

### YOLOv8 Nano Specifications

- **Parameters**: ~3.2M
- **FLOPs**: ~8.7B
- **Speed**: ~1-2ms per image (GPU)
- **Architecture**: CSPDarknet53 backbone + PANet neck + Detection head

### Training Hyperparameters

```python
{
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'optimizer': 'AdamW',
    'lr0': 0.01,              # Initial learning rate
    'lrf': 0.01,              # Final learning rate
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 50,           # Early stopping
    'cos_lr': True,           # Cosine LR scheduler
    'mosaic': 1.0,            # Mosaic augmentation
    'mixup': 0.0,
    'hsv_h': 0.015,           # HSV augmentation
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
}
```

### Loss Function

YOLOv8 uses a composite loss:

- **Box loss**: CIoU (Complete IoU) - measures bbox accuracy
- **Classification loss**: Binary cross-entropy - measures class prediction
- **DFL loss**: Distribution Focal Loss - refines bbox regression

## 🎯 Inference Configuration

### Detection Parameters

- **Confidence threshold**: 0.25 (adjustable)
  - Lower → more detections (higher recall, lower precision)
  - Higher → fewer false positives (higher precision, lower recall)
- **IoU threshold**: 0.45 (NMS)
  - Controls overlap between detections

### Adjusting Thresholds

Edit in `inference.py`:

```python
conf_threshold = 0.25  # Try 0.15 for more detections, 0.35 for fewer
iou_threshold = 0.45   # Standard NMS threshold
```

## ⚠️ Assumptions & Limitations

### Assumptions

1. **Visual similarity**: Test crabs visually resemble reference crabs
2. **Single class**: All crabs treated as one class (no species distinction)
3. **Static images**: Not optimized for video/temporal consistency
4. **Lighting**: Model trained on synthetic lighting may struggle with extreme underwater conditions
5. **Occlusion**: Heavily overlapping crabs may be missed or merged

### Limitations of Synthetic Training

| Limitation             | Impact                                         | Mitigation                                       |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------ |
| **Domain gap**         | Synthetic textures differ from real            | Pre-trained weights + color augmentation         |
| **Background bias**    | Random backgrounds may not match real seafloor | Use real underwater images as backgrounds        |
| **Pose diversity**     | Limited by 3 reference poses                   | Add more reference images if available           |
| **Occlusion handling** | Synthetic crabs rarely overlap heavily         | Generate more crowded scenes                     |
| **Scale distribution** | Uniform scale may not match real distribution  | Bias scale distribution based on camera distance |

### Known Issues

1. **Small crabs** (<30px): May be missed due to image size limitations
2. **Partial crabs**: Edge cases may be missed (crop augmentation could help)
3. **Similar objects**: Rocks/debris may cause false positives
4. **Lighting extremes**: Very dark/bright images may fail

## 🔧 Improving Performance

### Short-term Improvements (No New Data)

1. **Hyperparameter tuning**:

   ```bash
   # Try different confidence thresholds
   conf_threshold = 0.15  # More detections

   # Try larger model
   model_size = 's'  # YOLOv8s (more accurate, slower)
   ```

2. **More training data**:

   ```python
   # In generate_dataset.py
   num_train = 2000  # Increase from 1000
   num_val = 400     # Increase from 200
   ```

3. **Longer training**:
   ```python
   # In train.py
   epochs = 200      # Increase from 100
   patience = 100    # Increase early stopping patience
   ```

### Medium-term Improvements (With Real Data)

1. **Fine-tuning on real images**:

   - Manually annotate 50-100 real images
   - Fine-tune model: `model.train(data='real_data.yaml')`

2. **Active learning**:

   - Run inference on unlabeled images
   - Manually correct/confirm high-confidence predictions
   - Re-train with corrected labels

3. **Hard negative mining**:
   - Collect false positive images
   - Add to training set with negative examples

### Long-term Improvements (Production System)

1. **Real background images**: Use actual underwater environments
2. **Domain adaptation**: Use CycleGAN or similar to make synthetic images more realistic
3. **Ensemble models**: Combine multiple model sizes
4. **Temporal consistency**: Track detections across video frames
5. **Multi-stage detector**: Separate classifier after detection

## 📈 Expected Performance

### On Synthetic Validation Set

- **mAP@0.5**: 0.85-0.95 (expected after training)
- **mAP@0.5:0.95**: 0.60-0.75

### On Real Test Images

- **Best case** (similar to reference): 70-85% recall
- **Average case** (different lighting/scale): 50-70% recall
- **Worst case** (very different conditions): 30-50% recall

Performance heavily depends on:

- Visual similarity to reference images
- Image quality and lighting
- Crab size and occlusion
- Background complexity

## 🛠️ Troubleshooting

### Issue: Model not training / Loss is NaN

**Solution**: Reduce learning rate or batch size

```python
lr0 = 0.001  # Lower learning rate
batch_size = 8  # Smaller batches
```

### Issue: Out of GPU memory

**Solution**: Reduce batch size

```python
batch_size = 8  # Or 4 for very limited GPU
```

### Issue: No detections on test images

**Solution**:

1. Lower confidence threshold: `conf_threshold = 0.15`
2. Check if model trained successfully (view `results.png`)
3. Verify test images are similar to reference images

### Issue: Too many false positives

**Solution**:

1. Increase confidence threshold: `conf_threshold = 0.40`
2. Train longer for better precision
3. Add hard negative examples to training set

### Issue: Background removal failing

**Solution**:

1. Ensure reference images have clear backgrounds
2. Manually edit reference images to remove background
3. Use pre-segmented images if available

## 📚 References

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLO Paper**: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- **Transfer Learning**: Pan & Yang, "A Survey on Transfer Learning"
- **Synthetic Data**: Tremblay et al., "Training Deep Networks with Synthetic Data"

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{crab_detector_2026,
  author = {Your Name},
  title = {Crab Object Detection from Minimal Data},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/crab_classifier}
}
```

## 📄 License

MIT License - Feel free to use for academic or commercial purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Better background generation
- Domain adaptation techniques
- Real-time video inference
- Multi-species classification
- Deployment optimization (TensorRT, ONNX)

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy Crab Detecting! 🦀**
