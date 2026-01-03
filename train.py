"""
YOLOv8 Training Script for Crab Detection
=========================================

This script trains a YOLOv8 object detector on the synthetic crab dataset.

Model Choice: YOLOv8 Nano (yolov8n.pt)
--------------------------------------
YOLOv8n is ideal for this task because:

1. **Lightweight & Fast**: Suitable for deployment on edge devices (e.g., ROV systems)
2. **Transfer Learning**: Pre-trained on COCO dataset provides strong feature extractors
3. **Small Dataset Performance**: Works well with limited data thanks to pre-training
4. **Built-in Augmentation**: Includes mosaic, mixup, and other augmentations during training
5. **Multi-object Detection**: Naturally handles multiple crabs per image
6. **Easy Integration**: Simple API from Ultralytics

Why this is appropriate for synthetic data:
- Pre-trained weights help bridge the synthetic-to-real domain gap
- Built-in augmentation during training adds more variation
- Strong inductive bias from pre-training on natural images

Hyperparameters:
- Image size: 640x640 (standard for YOLOv8, good balance of accuracy/speed)
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 100 (with early stopping patience=50)
- Learning rate: 0.01 (default, uses cosine decay)
- Optimizer: AdamW (default in YOLOv8)
- Augmentation: YOLOv8 defaults (mosaic, mixup, HSV, flip, scale, translate)
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import yaml


def train_crab_detector(
    data_yaml='dataset/data.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device='',  # '' for auto-select, 'cpu', '0', '0,1' for GPUs
    project='runs/detect',
    name='crab_detector',
    exist_ok=False,
    pretrained=True,
    patience=50,
    save_period=10,
    workers=8
):
    """
    Train YOLOv8 model for crab detection.

    Args:
        data_yaml: Path to dataset configuration file
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to train on
        project: Project directory for saving runs
        name: Run name
        exist_ok: Whether to overwrite existing run
        pretrained: Whether to use pretrained weights
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        workers: Number of dataloader workers
    """

    print("="*80)
    print("CRAB DETECTION MODEL TRAINING")
    print("="*80)

    # Verify dataset exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"\n❌ Error: Dataset configuration not found at {data_yaml}")
        print("Please run generate_dataset.py first to create the synthetic dataset.")
        return

    # Load dataset configuration
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\n📊 Dataset Configuration:")
    print(f"   Path: {data_config['path']}")
    print(f"   Classes: {data_config['names']}")
    print(f"   Number of classes: {data_config['nc']}")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        device = 'cuda'
    elif torch.backends.mps.is_available():
        print(f"\n🚀 MPS (Apple Silicon) Available")
        device = 'mps'
    else:
        print(f"\n⚠️  No GPU detected, training on CPU (this will be slow)")
        device = 'cpu'

    print(f"\n🔧 Training Configuration:")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {device}")
    print(f"   Early stopping patience: {patience}")

    # Check if resuming from previous training
    best_weights_path = Path('weights/best.pt')
    if best_weights_path.exists():
        print(f"\n📦 Found existing checkpoint: {best_weights_path}")
        print("   🔄 Resuming training from weights/best.pt")
        model_name = str(best_weights_path)
    else:
        print(f"\n📦 No existing checkpoint found")
        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        print(f"   Starting fresh training with: {model_name}")

    print(f"\n📦 Loading model: {model_name}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("   ✓ Model loaded successfully")

    # Train the model
    print(f"\n🎯 Starting training...")
    print("="*80)

    try:
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            exist_ok=exist_ok,
            pretrained=pretrained,
            patience=patience,
            save_period=save_period,
            workers=workers,
            # Augmentation settings (using YOLOv8 defaults)
            # These complement our synthetic data generation
            hsv_h=0.015,        # HSV-Hue augmentation
            hsv_s=0.7,          # HSV-Saturation augmentation
            hsv_v=0.4,          # HSV-Value augmentation
            # Rotation (we already rotate in synthetic data)
            degrees=0.0,
            translate=0.1,      # Translation
            scale=0.5,          # Scale variation
            shear=0.0,          # Shear
            perspective=0.0,    # Perspective
            flipud=0.0,         # Vertical flip (we already do this)
            fliplr=0.5,         # Horizontal flip
            mosaic=1.0,         # Mosaic augmentation (very effective)
            mixup=0.0,          # Mixup augmentation
            copy_paste=0.0,     # Copy-paste augmentation
            # Optimizer settings
            optimizer='AdamW',  # Optimizer
            lr0=0.01,           # Initial learning rate
            lrf=0.01,           # Final learning rate (lr0 * lrf)
            momentum=0.937,     # SGD momentum/Adam beta1
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3.0,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            # Loss weights
            box=7.5,            # Box loss weight
            cls=0.5,            # Class loss weight
            dfl=1.5,            # DFL loss weight
            # Other settings
            verbose=True,       # Verbose output
            seed=42,            # Random seed for reproducibility
            deterministic=True,  # Deterministic mode
            single_cls=False,   # Multi-class dataset (3 crab species)
            rect=False,         # Rectangular training
            cos_lr=True,        # Cosine learning rate scheduler
            close_mosaic=10,    # Disable mosaic in last N epochs
            # Automatic Mixed Precision training
            amp=False if device in ["mps", "cpu"] else True,
            fraction=1.0,       # Dataset fraction to train on
            profile=False,      # Profile ONNX and TensorRT speeds
            freeze=None,        # Freeze layers (None or list of layer indices)

            agnostic_nms=True,  # Class-agnostic NMS
        )

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)

        # Print training results
        print(f"\n📈 Training Results:")
        print(
            f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(
            f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        # Get save paths
        save_dir = Path(project) / name
        best_model = save_dir / 'weights' / 'best.pt'
        last_model = save_dir / 'weights' / 'last.pt'

        print(f"\n💾 Model Checkpoints:")
        print(f"   Best model: {best_model}")
        print(f"   Last model: {last_model}")
        print(f"   Training plots: {save_dir}")

        print(f"\n🎯 Next Steps:")
        print(f"   1. Review training plots in: {save_dir}")
        print(f"   2. Run inference: python inference.py")
        print(f"   3. Validate on test images")

        print("="*80)

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main training function with default hyperparameters.

    Adjust these parameters based on your hardware and requirements:
    - Reduce batch_size if you run out of GPU memory
    - Increase epochs for potentially better results
    - Use larger model_size ('s', 'm', 'l') for better accuracy (slower)
    - Use smaller model_size ('n') for faster inference (lower accuracy)
    """

    train_crab_detector(
        data_yaml='dataset/data.yaml',
        model_size='n',          # Nano model - fast and lightweight
        epochs=100,              # 100 epochs with early stopping
        batch_size=16,           # Adjust based on GPU memory
        img_size=640,            # Standard YOLO input size
        device='mps',               # Auto-select device
        project='runs/detect',   # Output directory
        name='crab_detector',    # Run name
        exist_ok=False,          # Don't overwrite existing runs
        pretrained=True,         # Use COCO pre-trained weights
        patience=50,             # Early stopping patience
        save_period=10,          # Save checkpoint every 10 epochs
        workers=8                # Number of dataloader workers
    )


if __name__ == '__main__':
    main()
