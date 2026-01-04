"""
Training Script
==============

CLI wrapper for training YOLOv8 crab detection models.
"""

from models import CrabTrainer
from pathlib import Path
import yaml
import argparse


def main():
    """Main training function with default hyperparameters."""

    data_yaml = 'dataset/data.yaml'

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 crab detection model")
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
        model_size='n',  # Nano model
        device=''  # Auto-select
    )

    print(f"\n🔧 Training Configuration:")
    print(f"   Model: YOLOv8n (Nano)")
    print(f"   Image size: 640x640")
    print(f"   Batch size: 16")
    print(f"   Epochs: 100")
    print(f"   Device: {trainer.device}")
    print(f"   Early stopping patience: 50")

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
            epochs=100,
            batch_size=16,
            img_size=640,
            project='runs/detect',
            name='crab_detector',
            exist_ok=False,
            pretrained=True,
            patience=50,
            save_period=10,
            workers=8,
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
