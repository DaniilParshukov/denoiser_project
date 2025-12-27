import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DenoiseMaskDataset(Dataset):
    """
    Датасет для обучения модели денойзинга с масками
    Требует три типа данных:
    1. Noisy - зашумленное изображение
    2. Clean - чистое изображение (цель для денойзинга)
    3. Mask - бинарная маска (цель для векторизации)
    """
    
    def __init__(self, data_dir, mode='train', patch_size=128, synthetic_masks=True):
        """
        Args:
            data_dir: корневая директория с данными
            mode: 'train' или 'test'
            patch_size: размер патча для обучения
            synthetic_masks: генерировать маски синтетически если нет реальных
        """
        self.mode = mode
        self.patch_size = patch_size
        self.synthetic_masks = synthetic_masks
        
        # Пути к данным
        self.noisy_dir = os.path.join(data_dir, mode, 'noisy')
        self.clean_dir = os.path.join(data_dir, mode, 'clean')
        self.mask_dir = os.path.join(data_dir, mode, 'mask') if os.path.exists(os.path.join(data_dir, mode, 'mask')) else None
        
        # Получаем список файлов
        self.noisy_files = sorted([f for f in os.listdir(self.noisy_dir) 
                                  if f.endswith(('.png', '.jpg', '.bmp', '.tiff'))])
        
        print(f"Found {len(self.noisy_files)} images in {mode} set")
        
        # Аугментации для обучения
        if mode == 'train':
            self.transform = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.noisy_files)
    
    def generate_mask_from_clean(self, clean_img):
        """
        Генерация маски из чистого изображения
        Используется если нет реальных масок
        """
        # Бинаризация с адаптивным порогом
        if len(clean_img.shape) == 3:
            clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        else:
            clean_gray = clean_img
        
        # Адаптивный порог для получения маски
        mask = cv2.adaptiveThreshold(
            clean_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Морфологические операции для очистки
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Нормализация к [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        return mask
    
    def __getitem__(self, idx):
        # Загрузка noisy изображения
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        
        # Загрузка clean изображения (если есть)
        clean_path = os.path.join(self.clean_dir, self.noisy_files[idx])
        if os.path.exists(clean_path):
            clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Если нет clean, создаем его из noisy (для тестов)
            clean = noisy.copy()
        
        # Загрузка или генерация маски
        if self.mask_dir and os.path.exists(os.path.join(self.mask_dir, self.noisy_files[idx])):
            # Загружаем реальную маску
            mask_path = os.path.join(self.mask_dir, self.noisy_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            # Генерируем маску из clean изображения
            mask = self.generate_mask_from_clean(clean)
        
        # Преобразование к 3-канальному для albumentations
        noisy_3ch = np.stack([noisy] * 3, axis=-1)
        clean_3ch = np.stack([clean] * 3, axis=-1)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        # Применяем аугментации
        if self.mode == 'train':
            transformed = self.transform(image=noisy_3ch, 
                                         clean=clean_3ch, 
                                         mask=mask_3ch)
            
            noisy_tensor = transformed['image'][0:1, :, :]  # Берем только первый канал
            clean_tensor = transformed['clean'][0:1, :, :]
            mask_tensor = transformed['mask'][0:1, :, :]
        else:
            # Для теста сохраняем оригинальный размер
            transformed = self.transform(image=noisy_3ch)
            noisy_tensor = transformed['image'][0:1, :, :]
            
            # Clean и mask без изменения размера
            clean_tensor = torch.FloatTensor(clean).unsqueeze(0) / 255.0
            clean_tensor = (clean_tensor - 0.5) / 0.5  # Нормализация
            
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'mask': mask_tensor,
            'filename': self.noisy_files[idx]
        }


class SyntheticNoiseMaskDataset(Dataset):
    """
    Датасет для синтетической генерации данных
    Генерирует noisy, clean и mask на лету
    """
    
    def __init__(self, num_samples=1000, patch_size=128, 
                 noise_levels=[15, 25, 50], shapes=['lines', 'circles', 'rectangles']):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_levels = noise_levels
        self.shapes = shapes
        
        self.transform = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def generate_clean_image(self):
        """Генерация чистого изображения с геометрическими фигурами"""
        img = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        
        # Генерируем случайные фигуры
        num_shapes = random.randint(3, 8)
        
        for _ in range(num_shapes):
            shape_type = random.choice(self.shapes)
            
            if shape_type == 'lines':
                # Линии
                thickness = random.randint(1, 3)
                pt1 = (random.randint(0, self.patch_size), 
                       random.randint(0, self.patch_size))
                pt2 = (random.randint(0, self.patch_size), 
                       random.randint(0, self.patch_size))
                cv2.line(img, pt1, pt2, 1.0, thickness)
            
            elif shape_type == 'circles':
                # Круги
                radius = random.randint(5, 20)
                center = (random.randint(radius, self.patch_size - radius),
                          random.randint(radius, self.patch_size - radius))
                thickness = random.choice([-1, 1, 2])
                cv2.circle(img, center, radius, 1.0, thickness)
            
            elif shape_type == 'rectangles':
                # Прямоугольники
                pt1 = (random.randint(0, self.patch_size - 20),
                       random.randint(0, self.patch_size - 20))
                pt2 = (pt1[0] + random.randint(10, 40),
                       pt1[1] + random.randint(10, 40))
                thickness = random.choice([-1, 1, 2])
                cv2.rectangle(img, pt1, pt2, 1.0, thickness)
        
        return img
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Генерируем чистое изображение
        clean = self.generate_clean_image()
        
        # Маска = бинаризованное чистое изображение
        mask = (clean > 0).astype(np.float32)
        
        # Добавляем шум
        noise_level = random.choice(self.noise_levels) / 255.0
        noise = np.random.normal(0, noise_level, clean.shape)
        noisy = np.clip(clean + noise, 0, 1)
        
        # Конвертация в 3 канала
        noisy_3ch = np.stack([noisy * 255] * 3, axis=-1)
        clean_3ch = np.stack([clean * 255] * 3, axis=-1)
        mask_3ch = np.stack([mask * 255] * 3, axis=-1)
        
        # Применяем преобразования
        transformed = self.transform(image=noisy_3ch, 
                                     clean=clean_3ch,
                                     mask=mask_3ch)
        
        noisy_tensor = transformed['image'][0:1, :, :]
        clean_tensor = transformed['clean'][0:1, :, :]
        mask_tensor = transformed['mask'][0:1, :, :]
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'mask': mask_tensor,
            'filename': f'synthetic_{idx}.png'
        }


if __name__ == "__main__":
    # Тестирование датасета
    import matplotlib.pyplot as plt
    
    dataset = DenoiseMaskDataset('data', mode='train', patch_size=128)
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Денормализация для отображения
        noisy_img = (sample['noisy'].numpy().squeeze() * 0.5 + 0.5)
        clean_img = (sample['clean'].numpy().squeeze() * 0.5 + 0.5)
        mask_img = sample['mask'].numpy().squeeze()
        
        axes[0].imshow(noisy_img, cmap='gray')
        axes[0].set_title('Noisy')
        axes[0].axis('off')
        
        axes[1].imshow(clean_img, cmap='gray')
        axes[1].set_title('Clean')
        axes[1].axis('off')
        
        axes[2].imshow(mask_img, cmap='gray')
        axes[2].set_title('Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Noisy shape: {sample['noisy'].shape}")
        print(f"Clean shape: {sample['clean'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Mask range: [{mask_img.min():.3f}, {mask_img.max():.3f}]")
    else:
        print("No data found. Creating synthetic dataset...")
        syn_dataset = SyntheticNoiseMaskDataset(num_samples=10)
        sample = syn_dataset[0]
        print(f"Synthetic sample shapes: {sample['noisy'].shape}, {sample['clean'].shape}, {sample['mask'].shape}")