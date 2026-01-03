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
├── reference_images/          # Original reference images
├── test_images/               # Test images for evaluation
├── dataset/                   # Generated synthetic dataset
├── runs/                      # Training outputs
├── detections/                # Inference results
├── generate_dataset.py        # Synthetic data generation script
├── visualize_dataset.py       # Dataset visualization tool
├── train.py                   # Model training script
├── inference.py               # Static image inference script
├── live_inference.py          # Webcam/Video inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

Create a large dataset from the reference images.

```bash
python generate_dataset.py
```

- Segments crabs from `reference_images/`.
- Applies rotation, scaling, lighting, and noise augmentations.
- Generates 1000 training and 200 validation images in `dataset/`.

### 3. Visualize Dataset (Optional)

Verify the quality of generated data and annotations.

```bash
# View a grid of 16 random samples
python visualize_dataset.py --random --num_samples 16

# Analyze dataset statistics
python visualize_dataset.py --analyze
```

### 4. Train the Model

Train YOLOv8n on the synthetic dataset.

```bash
python train.py
```

- **Default**: YOLOv8 Nano, 100 epochs, image size 640.
- **Output**: Best weights saved to `runs/detect/crab_detector/weights/best.pt`.

### 5. Running Inference

#### Static Images
Detect crabs in images from `test_images/`.

```bash
python inference.py
```

- Results saved to `detections/`.
- Configuration: Edit `inference.py` to change thresholds.

#### Live Webcam
Run real-time detection on your webcam.

```bash
python live_inference.py
```

- **Controls**:
    - `Q` / `ESC`: Quit
    - `S`: Save current frame
    - `+` / `-`: Adjust confidence threshold
    - `F`: Toggle FPS display

- **Options**:
    ```bash
    # Specify different camera or model
    python live_inference.py --camera 1 --model runs/detect/crab_detector/weights/best.pt
    ```

## 🔧 Troubleshooting

- **Dataset not found**: Run `generate_dataset.py` first.
- **Model not found**: Run `train.py` first to generate the weights.
- **Out of Memory**: Decrease `batch_size` in `train.py` or use a smaller `model_size`.
- **Camera error**: Ensure no other application is using the webcam and try a different `camera_id` (e.g., `--camera 1`) in `live_inference.py`.

## 📚 References

- **YOLOv8**: [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- **RemBG**: [Rembg GitHub](https://github.com/danielgatis/rembg)
