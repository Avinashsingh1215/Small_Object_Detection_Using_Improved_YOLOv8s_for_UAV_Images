import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')

from utils.ema import EMA
from utils.plots import plot_training_curves

def train(cfg_path, resume_from=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Auto-resume
    if resume_from is None:
        project = cfg.get('project', 'runs/detect')
        name = cfg.get('name', 'train')
        last_ckpt = Path(project) / name / 'weights' / 'last.pt'
        if last_ckpt.exists():
            resume_from = str(last_ckpt)
            cfg['resume'] = True
            print(f"🔄 Resuming from {resume_from}")
        else:
            cfg['resume'] = False
    else:
        cfg['resume'] = True

    # Load model (standard YOLOv8s)
    if cfg['resume']:
        model = YOLO(resume_from)
    else:
        model = YOLO('yolov8s.pt')

    # Initialize EMA
    ema = EMA(model.model, decay=cfg.get('ema_decay', 0.9999))

    # Define callbacks
    def on_train_epoch_end(trainer):
        # Update EMA after each epoch
        ema.update()
        # Live metrics
        epoch = trainer.epoch + 1
        metrics = trainer.metrics
        if metrics:
            print(f"\n📊 Epoch {epoch}/{cfg['epochs']}")
            print(f"   Loss: {metrics.get('train/loss', 0):.4f}")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"   Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 0):.4f}")

    def on_train_end(trainer):
        # Apply EMA weights for final evaluation
        ema.apply_shadow()
        save_dir = Path(cfg.get('project', 'runs/detect')) / cfg.get('name', 'train') / 'weights'
        save_dir.mkdir(parents=True, exist_ok=True)
        ema_path = save_dir / 'ema_best.pt'
        torch.save(model.model.state_dict(), ema_path)
        print(f"✅ EMA weights saved to {ema_path}")
        ema.restore()
        # Generate plots
        print("\n📈 Generating plots...")
        csv_path = Path(cfg.get('project', 'runs/detect')) / cfg.get('name', 'train') / 'results.csv'
        if csv_path.exists():
            plot_training_curves(str(csv_path), str(csv_path.parent))

    # Register callbacks (Ultralytics method)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_train_end', on_train_end)

    # Filter out custom arguments before passing to YOLO
    custom_keys = [
        'multi_scale', 'scale_range', 'use_lightweight_pmse',
        'use_coord_attention', 'use_dynamic_assigner', 'ema_decay'
    ]
    yolo_cfg = {k: v for k, v in cfg.items() if k not in custom_keys}

    # Train
    results = model.train(**yolo_cfg)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/my_config.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    train(args.config, args.resume)
