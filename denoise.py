import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from models.dncnn import DnCNNWithMask, SimpleDnCNNMask
from utils.metrics import DenoiseMetrics


class ImageDenoiserWithMask:
    """Класс для денойзинга изображений с получением маски"""
    
    def __init__(self, model_path, model_type='simple', device=None):
        """
        Args:
            model_path: путь к весам модели
            model_type: 'simple' или 'dncnn'
            device: устройство для вычислений
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Загрузка модели
        if model_type == 'simple':
            self.model = SimpleDnCNNMask(channels=1)
        else:
            self.model = DnCNNWithMask(depth=7, n_channels=64, image_channels=1)
        
        # Загрузка весов
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_image(self, image):
        """
        Предобработка изображения для модели
        """
        if isinstance(image, str):
            # Загрузка из файла
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Сохраняем оригинальные размеры
        original_height, original_width = image.shape[:2]
        
        # Нормализация [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        
        # Паддинг до кратности 8 (для некоторых архитектур)
        pad_h = (8 - image.shape[0] % 8) % 8
        pad_w = (8 - image.shape[1] % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Преобразование в тензор [1, 1, H, W]
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Нормализация [-1, 1] как при обучении
        image_tensor = (image_tensor - 0.5) / 0.5
        
        return image_tensor, original_height, original_width, pad_h, pad_w
    
    def postprocess_result(self, denoised_tensor, mask_tensor, 
                          original_height, original_width, 
                          pad_h, pad_w, threshold=0.5):
        """
        Постобработка результатов модели
        """
        # Денормализация [-1, 1] -> [0, 1]
        denoised = denoised_tensor * 0.5 + 0.5
        
        # Убираем паддинг
        if pad_h > 0 or pad_w > 0:
            denoised = denoised[:, :, :original_height, :original_width]
            mask_tensor = mask_tensor[:, :, :original_height, :original_width]
        
        # Конвертация в numpy
        denoised_np = denoised.squeeze().cpu().numpy()
        mask_prob_np = mask_tensor.squeeze().cpu().numpy()
        
        # Бинаризация маски
        mask_binary_np = (mask_prob_np > threshold).astype(np.float32)
        
        # Конвертация в uint8 для сохранения
        denoised_uint8 = (denoised_np * 255).astype(np.uint8)
        mask_prob_uint8 = (mask_prob_np * 255).astype(np.uint8)
        mask_binary_uint8 = (mask_binary_np * 255).astype(np.uint8)
        
        return {
            'denoised': denoised_uint8,
            'mask_prob': mask_prob_uint8,
            'mask_binary': mask_binary_uint8,
            'mask_prob_raw': mask_prob_np,
            'mask_binary_raw': mask_binary_np
        }
    
    def denoise_image(self, image, threshold=0.5):
        """
        Основная функция денойзинга с получением маски
        """
        # Предобработка
        image_tensor, orig_h, orig_w, pad_h, pad_w = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Инференс
        with torch.no_grad():
            denoised_tensor, mask_tensor = self.model(image_tensor)
        
        # Постобработка
        results = self.postprocess_result(
            denoised_tensor, mask_tensor,
            orig_h, orig_w, pad_h, pad_w,
            threshold
        )
        
        return results
    
    def process_folder(self, input_dir, output_dir, threshold=0.5, 
                       save_all=True, image_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        """
        Обработка всех изображений в папке
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Создание подпапок
        output_path.mkdir(parents=True, exist_ok=True)
        if save_all:
            (output_path / 'denoised').mkdir(exist_ok=True)
            (output_path / 'mask_prob').mkdir(exist_ok=True)
            (output_path / 'mask_binary').mkdir(exist_ok=True)
            (output_path / 'combined').mkdir(exist_ok=True)
        
        # Поиск изображений
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        results = []
        for img_path in image_files:
            try:
                print(f"Processing {img_path.name}...")
                
                # Денойзинг
                results_dict = self.denoise_image(str(img_path), threshold)
                
                # Сохранение результатов
                stem = img_path.stem
                
                if save_all:
                    cv2.imwrite(str(output_path / 'denoised' / f'{stem}_denoised.png'), 
                               results_dict['denoised'])
                    cv2.imwrite(str(output_path / 'mask_prob' / f'{stem}_mask_prob.png'), 
                               results_dict['mask_prob'])
                    cv2.imwrite(str(output_path / 'mask_binary' / f'{stem}_mask_binary.png'), 
                               results_dict['mask_binary'])
                
                # Сохранение комбинированного изображения
                self.save_combined_visualization(
                    str(img_path), results_dict,
                    str(output_path / 'combined' / f'{stem}_combined.png')
                )
                
                results.append({
                    'filename': img_path.name,
                    'denoised': results_dict['denoised'],
                    'mask_prob': results_dict['mask_prob'],
                    'mask_binary': results_dict['mask_binary']
                })
                
                print(f"  ✓ Saved results for {img_path.name}")
                
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
        
        return results
    
    def save_combined_visualization(self, input_path, results_dict, output_path):
        """
        Создание комбинированной визуализации
        """
        # Загрузка оригинального изображения
        original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Создание фигуры
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Оригинальное
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original (Noisy)')
        axes[0, 0].axis('off')
        
        # Денойзинг результат
        axes[0, 1].imshow(results_dict['denoised'], cmap='gray')
        axes[0, 1].set_title('Denoised')
        axes[0, 1].axis('off')
        
        # Маска (вероятность)
        axes[0, 2].imshow(results_dict['mask_prob'], cmap='gray')
        axes[0, 2].set_title('Mask (Probability)')
        axes[0, 2].axis('off')
        
        # Маска (бинарная)
        axes[1, 0].imshow(results_dict['mask_binary'], cmap='gray')
        axes[1, 0].set_title('Mask (Binary)')
        axes[1, 0].axis('off')
        
        # Наложение денойзинга на оригинал (разница)
        diff = cv2.absdiff(original, results_dict['denoised'])
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title('Difference (Removed Noise)')
        axes[1, 1].axis('off')
        
        # Наложение маски на денойзинг
        masked = cv2.bitwise_and(results_dict['denoised'], results_dict['denoised'], 
                                mask=results_dict['mask_binary'])
        axes[1, 2].imshow(masked, cmap='gray')
        axes[1, 2].set_title('Denoised + Mask')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Denoise images and extract masks')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'dncnn'],
                       help='Model type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--save_all', action='store_true',
                       help='Save all intermediate results')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Определение устройства
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание денойзера
    denoiser = ImageDenoiserWithMask(args.model, args.model_type, device)
    
    # Проверка: файл или папка
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Обработка одного файла
        print(f"Processing single image: {args.input}")
        results = denoiser.denoise_image(args.input, args.threshold)
        
        # Сохранение
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stem = input_path.stem
        cv2.imwrite(str(output_path / f'{stem}_denoised.png'), results['denoised'])
        cv2.imwrite(str(output_path / f'{stem}_mask_binary.png'), results['mask_binary'])
        
        # Визуализация
        denoiser.save_combined_visualization(
            args.input, results,
            str(output_path / f'{stem}_combined.png')
        )
        
        print(f"Results saved to {args.output}")
        
    elif input_path.is_dir():
        # Обработка папки
        print(f"Processing directory: {args.input}")
        results = denoiser.process_folder(
            args.input, args.output, 
            args.threshold, args.save_all
        )
        
        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {args.output}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()