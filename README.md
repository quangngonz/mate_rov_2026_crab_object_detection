# Crab Object Detection System

A complete pipeline for detecting multiple crabs in images, supporting synthetic data generation, training, and both static and live inference.

## 🎯 Project Overview

This system demonstrates how to build a robust object detector starting from minimal data (3 reference images).

- **Input**: Reference images of crabs.
- **Process**: Generate synthetic training data with extensive augmentation.
- **Output**: YOLOv8 model capable of detecting multiple crabs.

## 📁 Directory Structure

```
crab_classifier/
├── config/              # Configuration files
├── data/                # Data processing logic
├── dataset/             # Generated synthetic dataset (train/val)
├── models/              # Model definitions and training logic
├── reference_images/    # Source images for synthetic generation
├── test_images/         # Images for testing inference
├── ui/                  # UI components for live inference
├── utils/               # Utility functions
├── weights/             # Pre-trained or best model weights
├── generate_dataset.py  # Script to create synthetic dataset
├── inference.py         # Script for static image inference
├── live_inference.py    # Script for real-time webcam inference
├── train.py             # Script to train the model
├── extract_and_label.py # Extract video frames and auto-label
├── review_and_train.py  # Review labels and fine-tune model
├── visualize_dataset.py # Tool to inspect the dataset
├── requirements.txt     # Python dependencies
└── VIDEO_WORKFLOW.md    # Video-based improvement guide
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

Create a synthetic dataset from the images in `reference_images/`.

```bash
python generate_dataset.py
```

### 3. Visualize Data (Optional)

Check the generated images to ensure quality.

```bash
# View 16 random samples
python visualize_dataset.py --random --num_samples 16

# Analyze dataset statistics
python visualize_dataset.py --analyze

# Visualize a single specific image
python visualize_dataset.py --single image_name.jpg --split train

# Save visualization to a file
python visualize_dataset.py --random --output visualization.png
```

### 4. Train Model

Train the YOLOv8n model on the generated dataset.

```bash
python train.py
```

Weights will be saved to `runs/detect/crab_detector/weights/best.pt`.

### 5. Inference

#### Static Images

Detect crabs in images located in `test_images/`. Results are saved to `detections/`.

```bash
python inference.py
```

#### Live Webcam

Run real-time detection on your camera feed.

```bash
python live_inference.py
```

**Controls:**

- `Q` / `ESC`: Quit
- `S`: Save current frame
- `+` / `-`: Adjust confidence threshold
- `F`: Toggle FPS display

**Live Inference Options:**

| Argument   | Default           | Description                      |
| :--------- | :---------------- | :------------------------------- |
| `--model`  | `weights/best.pt` | Path to model weights            |
| `--conf`   | `0.77`            | Confidence threshold (0.0 - 1.0) |
| `--iou`    | `0.45`            | NMS IoU threshold (0.0 - 1.0)    |
| `--camera` | `0`               | Camera device ID                 |
| `--width`  | `1920`            | Camera frame width               |
| `--height` | `1080`            | Camera frame height              |

**Example:**

```bash
python live_inference.py --camera 1 --conf 0.5 --width 1280 --height 720
```

## 🎥 Video-Based Model Improvement

Improve your model by extracting frames from underwater videos, auto-labeling them, and fine-tuning with corrected labels.

### Workflow Overview

1. **Extract & Auto-Label**: Extract frames from video and automatically label using current model
2. **Review & Correct**: Interactive UI to review and fix labels
3. **Fine-tune**: Train model with additional data

### Quick Example

```bash
# Step 1: Extract frames and auto-label
python extract_and_label.py your_video.mp4 --fps 1.0

# Step 2: Review labels and train
python review_and_train.py
# - Use arrow keys to navigate
# - Click & drag to add boxes
# - Press D to delete boxes
# - Press T to start training

# Step 3: Use improved model
cp runs/detect/crab_detector_finetuned/weights/best.pt weights/best.pt
```

📖 **For detailed instructions**, see [VIDEO_WORKFLOW.md](VIDEO_WORKFLOW.md)

## 🔧 Troubleshooting

- **Dataset not found**: Run `generate_dataset.py`.
- **Model not found**: Run `train.py` or ensure `weights/best.pt` exists.
- **Camera error**: Try a different `--camera` ID (e.g., `--camera 1`) or check if another app is using it.
