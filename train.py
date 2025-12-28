import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.dncnn import DnCNNWithMask, SimpleDnCNNMask
from utils.dataset import DenoiseMaskDataset, SyntheticNoiseMaskDataset
from utils.metrics import DenoiseMetrics


class MaskAwareLoss(nn.Module):
    """
    Комбинированная функция потерь для денойзинга и маски
    """
    def __init__(self, denoise_weight=1.0, mask_weight=1.0, 
                 denoise_loss='mse', mask_loss='bce'):
        super().__init__()
        self.denoise_weight = denoise_weight
        self.mask_weight = mask_weight
        
        # Функция потерь для денойзинга
        if denoise_loss == 'mse':
            self.denoise_loss_fn = nn.MSELoss()
        elif denoise_loss == 'l1':
            self.denoise_loss_fn = nn.L1Loss()
        elif denoise_loss == 'huber':
            self.denoise_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown denoise loss: {denoise_loss}")
        
        # Функция потерь для маски
        if mask_loss == 'bce':
            self.mask_loss_fn = nn.BCELoss()
        elif mask_loss == 'dice':
            self.mask_loss_fn = DiceLoss()
        elif mask_loss == 'bce_dice':
            self.mask_loss_fn = BCEDiceLoss()
        else:
            raise ValueError(f"Unknown mask loss: {mask_loss}")
    
    def forward(self, denoised_pred, clean_true, mask_pred, mask_true):
        denoise_loss = self.denoise_loss_fn(denoised_pred, clean_true)
        mask_loss = self.mask_loss_fn(mask_pred, mask_true)
        
        total_loss = (self.denoise_weight * denoise_loss + 
                     self.mask_weight * mask_loss)
        
        return {
            'total': total_loss,
            'denoise': denoise_loss,
            'mask': mask_loss
        }


class DiceLoss(nn.Module):
    """Dice Loss для сегментации"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Комбинированная BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Одна эпоха обучения"""
    model.train()
    running_loss = {'total': 0.0, 'denoise': 0.0, 'mask': 0.0}
    running_metrics = {'psnr': 0.0, 'ssim': 0.0, 'iou': 0.0, 'f1': 0.0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        mask_true = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        denoised_pred, mask_pred = model(noisy)
        
        # Вычисление потерь
        losses = criterion(denoised_pred, clean, mask_pred, mask_true)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Аккумулируем потери
        for key in running_loss:
            running_loss[key] += losses[key].item()
        
        # Вычисляем метрики на лету (раз в несколько батчей для скорости)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                # Берем первый элемент батча для метрик
                metrics = DenoiseMetrics.calculate_all_metrics(
                    denoised_pred[0:1], clean[0:1], 
                    mask_pred[0:1], mask_true[0:1]
                )
                
                for key in running_metrics:
                    if key in metrics:
                        running_metrics[key] += metrics[key]
        
        # Обновление progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'PSNR': f"{metrics.get('psnr', 0):.2f}",
                'IoU': f"{metrics.get('iou', 0):.3f}"
            })
        
        # Логирование в TensorBoard
        if writer and batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_total', losses['total'].item(), global_step)
            writer.add_scalar('Loss/train_denoise', losses['denoise'].item(), global_step)
            writer.add_scalar('Loss/train_mask', losses['mask'].item(), global_step)
    
    # Средние значения за эпоху
    avg_losses = {k: v / len(dataloader) for k, v in running_loss.items()}
    avg_metrics = {k: v / max(1, len(dataloader) // 10) for k, v in running_metrics.items()}
    
    return avg_losses, avg_metrics


def validate(model, dataloader, criterion, device, writer=None, epoch=0):
    """Валидация модели"""
    model.eval()
    running_loss = {'total': 0.0, 'denoise': 0.0, 'mask': 0.0}
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask_true = batch['mask'].to(device)
            
            denoised_pred, mask_pred = model(noisy)
            
            # Потери
            losses = criterion(denoised_pred, clean, mask_pred, mask_true)
            
            for key in running_loss:
                running_loss[key] += losses[key].item()
            
            # Метрики для каждого элемента батча
            for i in range(noisy.shape[0]):
                metrics = DenoiseMetrics.calculate_all_metrics(
                    denoised_pred[i:i+1], clean[i:i+1],
                    mask_pred[i:i+1], mask_true[i:i+1]
                )
                all_metrics.append(metrics)
    
    # Средние значения
    avg_losses = {k: v / len(dataloader) for k, v in running_loss.items()}
    
    # Средние метрики по всем примерам
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Логирование в TensorBoard
    if writer:
        writer.add_scalar('Loss/val_total', avg_losses['total'], epoch)
        writer.add_scalar('Loss/val_denoise', avg_losses['denoise'], epoch)
        writer.add_scalar('Loss/val_mask', avg_losses['mask'], epoch)
        
        for metric_name, metric_value in avg_metrics.items():
            writer.add_scalar(f'Metrics/val_{metric_name}', metric_value, epoch)
    
    return avg_losses, avg_metrics


def save_checkpoint(model, optimizer, epoch, losses, metrics, path):
    """Сохранение чекпоинта"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'metrics': metrics,
    }, path)


def visualize_results(model, dataloader, device, save_dir, num_samples=5):
    """Визуализация результатов модели"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            mask_true = batch['mask'].to(device)
            filename = batch['filename'][0]
            
            denoised_pred, mask_pred = model(noisy)
            
            # Конвертация в numpy для визуализации
            noisy_np = noisy[0].cpu().numpy().squeeze()
            clean_np = clean[0].cpu().numpy().squeeze()
            denoised_np = denoised_pred[0].cpu().numpy().squeeze()
            mask_true_np = mask_true[0].cpu().numpy().squeeze()
            mask_pred_np = mask_pred[0].cpu().numpy().squeeze()
            
            # Денормализация
            noisy_np = (noisy_np * 0.5 + 0.5)
            clean_np = (clean_np * 0.5 + 0.5)
            denoised_np = (denoised_np * 0.5 + 0.5)
            
            # Бинаризация маски
            mask_pred_binary = (mask_pred_np > 0.5).astype(np.float32)
            
            # Создание визуализации
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            images = [
                (noisy_np, 'Noisy Input'),
                (clean_np, 'Clean Ground Truth'),
                (denoised_np, 'Denoised Output'),
                (mask_true_np, 'True Mask'),
                (mask_pred_np, 'Predicted Mask (Prob)'),
                (mask_pred_binary, 'Predicted Mask (Binary)')
            ]
            
            for idx, (img, title) in enumerate(images):
                ax = axes[idx // 3, idx % 3]
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
            
            # Вычисление метрик для этого примера
            metrics = DenoiseMetrics.calculate_all_metrics(
                denoised_pred[0:1], clean[0:1],
                mask_pred[0:1], mask_true[0:1]
            )
            
            # Добавляем метрики в заголовок
            metrics_text = f"PSNR: {metrics['psnr']:.2f}dB, IoU: {metrics['iou']:.3f}, F1: {metrics['f1']:.3f}"
            plt.suptitle(f"{filename}\n{metrics_text}", fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'result_{filename}.png'), dpi=150, bbox_inches='tight')
            plt.close()


def main(args):
    """Основная функция обучения"""
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Создание директорий
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    # Модель
    if args.model == 'dncnn':
        model = DnCNNWithMask(
            depth=args.depth,
            n_channels=args.channels,
            image_channels=1,
            use_bnorm=not args.no_bn
        )
    elif args.model == 'simple':
        model = SimpleDnCNNMask(channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Датасеты
    if args.synthetic:
        print("Using synthetic dataset")
        train_dataset = SyntheticNoiseMaskDataset(
            num_samples=args.synthetic_samples,
            patch_size=args.patch_size,
            noise_levels=args.noise_levels
        )
    else:
        print(f"Using real dataset from {args.data_dir}")
        train_dataset = DenoiseMaskDataset(
            data_dir=args.data_dir,
            mode='train',
            patch_size=args.patch_size,
            synthetic_masks=args.synthetic_masks
        )
    
    val_dataset = DenoiseMaskDataset(
        data_dir=args.data_dir,
        mode='test',
        patch_size=args.patch_size,
        synthetic_masks=args.synthetic_masks
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Loss function
    criterion = MaskAwareLoss(
        denoise_weight=args.denoise_weight,
        mask_weight=args.mask_weight,
        denoise_loss=args.denoise_loss,
        mask_loss=args.mask_loss
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience
    )
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Обучение
    best_val_iou = 0.0
    train_history = {
        'loss_total': [], 'loss_denoise': [], 'loss_mask': [],
        'val_loss_total': [], 'val_loss_denoise': [], 'val_loss_mask': [],
        'psnr': [], 'val_psnr': [], 'iou': [], 'val_iou': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Обучение
        train_losses, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Валидация
        val_losses, val_metrics = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        
        # Обновление scheduler
        scheduler.step(val_losses['total'])
        
        # Сохранение истории
        for key in train_history:
            if key.startswith('val_'):
                base_key = key[4:]
                if base_key in val_losses:
                    train_history[key].append(val_losses[base_key])
                elif base_key in val_metrics:
                    train_history[key].append(val_metrics[base_key])
            else:
                if key in train_losses:
                    train_history[key].append(train_losses[key])
                elif key in train_metrics:
                    train_history[key].append(train_metrics[key])
        
        # Вывод результатов
        print(f"\nTrain - Loss: {train_losses['total']:.4f}, "
              f"PSNR: {train_metrics.get('psnr', 0):.2f}dB, "
              f"IoU: {train_metrics.get('iou', 0):.3f}")
        print(f"Val   - Loss: {val_losses['total']:.4f}, "
              f"PSNR: {val_metrics.get('psnr', 0):.2f}dB, "
              f"IoU: {val_metrics.get('iou', 0):.3f}")
        
        # Сохранение лучшей модели
        val_iou = val_metrics.get('iou', 0)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch, val_losses, val_metrics,
                os.path.join(args.checkpoint_dir, 'best_model.pth')
            )
            print(f"New best model! IoU: {best_val_iou:.3f}")
            
            # Визуализация результатов лучшей модели
            visualize_results(
                model, val_loader, device,
                os.path.join(args.visualize_dir, 'best_model'),
                num_samples=min(5, len(val_dataset))
            )
        
        # Регулярное сохранение
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_losses, val_metrics,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            
            # Визуализация
            visualize_results(
                model, val_loader, device,
                os.path.join(args.visualize_dir, f'epoch_{epoch}'),
                num_samples=min(3, len(val_dataset))
            )
    
    # Финальное сохранение
    save_checkpoint(
        model, optimizer, args.epochs, val_losses, val_metrics,
        os.path.join(args.checkpoint_dir, 'final_model.pth')
    )
    
    # Сохранение истории обучения
    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # Визуализация кривых обучения
    plot_training_history(train_history, args.log_dir)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation IoU: {best_val_iou:.3f}")
    print(f"Models saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print(f"Visualizations saved in: {args.visualize_dir}")
    print(f"{'='*60}")
    
    writer.close()


def plot_training_history(history, save_dir):
    """Визуализация процесса обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['loss_total'], label='Train Total Loss')
    axes[0, 0].plot(history['val_loss_total'], label='Val Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR
    axes[0, 1].plot(history['psnr'], label='Train PSNR')
    axes[0, 1].plot(history['val_psnr'], label='Val PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(history['iou'], label='Train IoU')
    axes[1, 0].plot(history['val_iou'], label='Val IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Mask IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Separate losses
    axes[1, 1].plot(history['loss_denoise'], label='Train Denoise Loss')
    axes[1, 1].plot(history['loss_mask'], label='Train Mask Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Component Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DnCNN with Mask')
    
    # Пути
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory with data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--visualize_dir', type=str, default='visualizations',
                       help='Directory for visualizations')
    
    # Параметры модели
    parser.add_argument('--model', type=str, default='simple',
                       choices=['dncnn', 'simple'],
                       help='Model architecture')
    parser.add_argument('--depth', type=int, default=7,
                       help='Number of layers for DnCNN')
    parser.add_argument('--channels', type=int, default=64,
                       help='Number of channels in hidden layers')
    parser.add_argument('--no_bn', action='store_true',
                       help='Disable batch normalization')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for LR scheduler')
    
    # Loss weights
    parser.add_argument('--denoise_weight', type=float, default=1.0,
                       help='Weight for denoising loss')
    parser.add_argument('--mask_weight', type=float, default=1.0,
                       help='Weight for mask loss')
    parser.add_argument('--denoise_loss', type=str, default='mse',
                       choices=['mse', 'l1', 'huber'],
                       help='Denoising loss function')
    parser.add_argument('--mask_loss', type=str, default='bce',
                       choices=['bce', 'dice', 'bce_dice'],
                       help='Mask loss function')
    
    # Датасет
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic dataset')
    parser.add_argument('--synthetic_samples', type=int, default=1000,
                       help='Number of synthetic samples')
    parser.add_argument('--synthetic_masks', action='store_true',
                       help='Generate masks synthetically if not available')
    parser.add_argument('--noise_levels', nargs='+', type=int,
                       default=[15, 25, 50],
                       help='Noise levels for synthetic noise')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch size for training')
    
    # Системные
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU even if GPU is available')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Создание директорий если не существуют
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)
    
    # Запуск обучения
    main(args)