import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F


class DenoiseMetrics:
    """Класс для вычисления метрик качества денойзинга и масок"""
    
    @staticmethod
    def calculate_psnr(img1, img2, data_range=1.0):
        """Вычисление PSNR между двумя изображениями"""
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        
        return psnr(img1, img2, data_range=data_range)
    
    @staticmethod
    def calculate_ssim(img1, img2, win_size=7):
        """Вычисление SSIM между двумя изображениями"""
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        
        # Для одноканальных изображений
        return ssim(img1, img2, win_size=win_size, data_range=1.0, channel_axis=None)
    
    @staticmethod
    def calculate_iou(mask_pred, mask_true, threshold=0.5):
        """
        Вычисление Intersection over Union для бинарных масок
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.detach().cpu().numpy()
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.detach().cpu().numpy()
        
        mask_pred = mask_pred.squeeze() > threshold
        mask_true = mask_true.squeeze() > 0.5
        
        intersection = np.logical_and(mask_pred, mask_true).sum()
        union = np.logical_or(mask_pred, mask_true).sum()
        
        if union == 0:
            return 1.0  # Обе маски пустые
        
        return intersection / union
    
    @staticmethod
    def calculate_dice(mask_pred, mask_true, threshold=0.5):
        """Вычисление Dice coefficient"""
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.detach().cpu().numpy()
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.detach().cpu().numpy()
        
        mask_pred = mask_pred.squeeze() > threshold
        mask_true = mask_true.squeeze() > 0.5
        
        intersection = np.logical_and(mask_pred, mask_true).sum()
        
        if intersection == 0:
            return 0.0
        
        return 2.0 * intersection / (mask_pred.sum() + mask_true.sum())
    
    @staticmethod
    def calculate_accuracy(mask_pred, mask_true, threshold=0.5):
        """Вычисление accuracy для бинарной классификации"""
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.detach().cpu().numpy()
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.detach().cpu().numpy()
        
        mask_pred = mask_pred.squeeze() > threshold
        mask_true = mask_true.squeeze() > 0.5
        
        correct = (mask_pred == mask_true).sum()
        total = mask_pred.size
        
        return correct / total
    
    @staticmethod
    def calculate_precision_recall(mask_pred, mask_true, threshold=0.5):
        """Вычисление precision и recall"""
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.detach().cpu().numpy()
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.detach().cpu().numpy()
        
        mask_pred = mask_pred.squeeze() > threshold
        mask_true = mask_true.squeeze() > 0.5
        
        true_pos = np.logical_and(mask_pred, mask_true).sum()
        false_pos = np.logical_and(mask_pred, np.logical_not(mask_true)).sum()
        false_neg = np.logical_and(np.logical_not(mask_pred), mask_true).sum()
        
        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        
        return precision, recall
    
    @staticmethod
    def calculate_f1(precision, recall):
        """Вычисление F1-score"""
        if precision + recall == 0:
            return 0.0
        return 2.0 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_all_metrics(denoised, clean, mask_pred, mask_true, threshold=0.5):
        """Вычисление всех метрик"""
        metrics = {}
        
        # Метрики денойзинга
        metrics['psnr'] = DenoiseMetrics.calculate_psnr(denoised, clean)
        metrics['ssim'] = DenoiseMetrics.calculate_ssim(denoised, clean)
        
        # Метрики маски
        metrics['iou'] = DenoiseMetrics.calculate_iou(mask_pred, mask_true, threshold)
        metrics['dice'] = DenoiseMetrics.calculate_dice(mask_pred, mask_true, threshold)
        metrics['accuracy'] = DenoiseMetrics.calculate_accuracy(mask_pred, mask_true, threshold)
        
        precision, recall = DenoiseMetrics.calculate_precision_recall(mask_pred, mask_true, threshold)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = DenoiseMetrics.calculate_f1(precision, recall)
        
        return metrics
    
    @staticmethod
    def mask_to_contour_metrics(mask_pred, mask_true, threshold=0.5):
        """
        Метрики для оценки качества контуров в маске
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.detach().cpu().numpy()
        if isinstance(mask_true, torch.Tensor):
            mask_true = mask_true.detach().cpu().numpy()
        
        mask_pred = (mask_pred.squeeze() > threshold).astype(np.uint8) * 255
        mask_true = (mask_true.squeeze() > 0.5).astype(np.uint8) * 255
        
        # Находим контуры
        contours_pred = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_true = cv2.findContours(mask_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Вычисляем площади
        area_pred = sum(cv2.contourArea(c) for c in contours_pred)
        area_true = sum(cv2.contourArea(c) for c in contours_true)
        
        # Количество контуров
        num_contours_pred = len(contours_pred)
        num_contours_true = len(contours_true)
        
        # Разница в площадях
        if area_true > 0:
            area_ratio = area_pred / area_true
        else:
            area_ratio = 1.0 if area_pred == 0 else float('inf')
        
        return {
            'num_contours_pred': num_contours_pred,
            'num_contours_true': num_contours_true,
            'area_pred': area_pred,
            'area_true': area_true,
            'area_ratio': area_ratio,
            'contour_count_ratio': num_contours_pred / max(num_contours_true, 1)
        }


if __name__ == "__main__":
    # Тестирование метрик
    import matplotlib.pyplot as plt
    
    # Создаем тестовые данные
    img_size = 128
    clean = np.zeros((img_size, img_size), dtype=np.float32)
    cv2.circle(clean, (64, 64), 30, 1.0, -1)
    
    # Добавляем шум
    noisy = clean + np.random.normal(0, 0.1, clean.shape)
    noisy = np.clip(noisy, 0, 1)
    
    # Идеальная маска
    true_mask = (clean > 0.5).astype(np.float32)
    
    # Предсказанная маска (с ошибками)
    pred_mask = true_mask.copy()
    # Добавляем шум в маску
    pred_mask = pred_mask + np.random.normal(0, 0.2, pred_mask.shape)
    pred_mask = np.clip(pred_mask, 0, 1)
    
    # Вычисляем метрики
    metrics = DenoiseMetrics.calculate_all_metrics(noisy, clean, pred_mask, true_mask)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key:15}: {value:.4f}")
    
    # Контурные метрики
    contour_metrics = DenoiseMetrics.mask_to_contour_metrics(pred_mask, true_mask)
    print("\nContour Metrics:")
    for key, value in contour_metrics.items():
        print(f"{key:25}: {value:.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('Clean')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(true_mask, cmap='gray')
    axes[1, 0].set_title('True Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_mask, cmap='gray')
    axes[1, 1].set_title(f'Pred Mask (IoU: {metrics["iou"]:.3f})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()