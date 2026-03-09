"""
Training Script
==============

CLI wrapper for training YOLOv26 crab detection models.
"""

from models import CrabTrainer
from pathlib import Path
import yaml
import argparse

from config.constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_SAVE_PERIOD,
    DEFAULT_WORKERS
)


def main():
    """Main training function with default hyperparameters."""

    data_yaml = 'dataset/data.yaml'

    parser = argparse.ArgumentParser(
        description="Train YOLOv26 crab detection model")
    parser.add_argument(
        '--use_last_weights', type=str, default='False',
        help="Whether to use last weights to resume training if available (True/False)"
    )
    args = parser.parse_args()

    print("="*80)
    print("CRAB DETECTION MODEL TRAINING")
    print("="*80)

    # Verify dataset exists
    if not Path(data_yaml).exists():
        print(f"\n❌ Error: Dataset configuration not found at {data_yaml}")
        print("Please run generate_dataset.py first to create the synthetic dataset.")
        return

    # Load and display dataset config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\n📊 Dataset Configuration:")
    print(f"   Path: {data_config['path']}")
    print(f"   Classes: {data_config['names']}")
    print(f"   Number of classes: {data_config['nc']}")

    # Initialize trainer
    trainer = CrabTrainer(
        data_yaml=data_yaml,
        model_size=DEFAULT_MODEL_SIZE,  # Nano model
        device=''  # Auto-select
    )

    print(f"\n🔧 Training Configuration:")
    print(f"   Model: YOLOv26{DEFAULT_MODEL_SIZE} (Nano)")
    print(f"   Image size: {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}")
    print(f"   Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"   Epochs: {DEFAULT_EPOCHS}")
    print(f"   Device: {trainer.device}")
    print(f"   Early stopping patience: {DEFAULT_PATIENCE} epochs")

    # Check for existing checkpoint
    best_weights = Path('weights/best.pt')
    resume_checkpoint = str(best_weights) if args.use_last_weights.lower(
    ) == 'true' and best_weights.exists() else None

    if resume_checkpoint:
        print(f"\n📦 Found existing checkpoint: {resume_checkpoint}")
        print("   🔄 Resuming training from checkpoint")
    else:
        print(f"\n📦 No existing checkpoint found")
        print("   Starting fresh training with pretrained weights")

    # Train
    print(f"\n🎯 Starting training...")
    print("="*80)

    try:
        results = trainer.train(
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            img_size=DEFAULT_IMAGE_SIZE[0],
            project='runs/detect',
            name='crab_detector',
            exist_ok=False,
            pretrained=True,
            patience=DEFAULT_PATIENCE,
            save_period=DEFAULT_SAVE_PERIOD,
            workers=DEFAULT_WORKERS,
            resume_checkpoint=resume_checkpoint
        )

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\n💾 Model Checkpoints:")
        print(f"   Best model: {results['best_model']}")
        print(f"   Last model: {results['last_model']}")
        print(f"   Training plots: {results['save_dir']}")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review training plots in: {results['save_dir']}")
        print(f"   2. Run inference: python scripts/inference.py")
        print(f"   3. Validate on test images")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
