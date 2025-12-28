import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import random
import glob
from PIL import Image


class DenoiseMaskDataset(Dataset):
    """
    Dataset для денойзинга с масками
    """
    def __init__(self, data_dir, mode='train', patch_size=128, 
                 synthetic_masks=False, transform=None):
        """
        Args:
            data_dir: корневая директория с данными
            mode: 'train' или 'test'
            patch_size: размер патча для обучения
            synthetic_masks: генерировать маски синтетически если нет настоящих
            transform: трансформации для аугментации
        """
        self.data_dir = Path(data_dir) / mode
        self.mode = mode
        self.patch_size = patch_size
        self.synthetic_masks = synthetic_masks
        self.transform = transform
        
        # Проверяем существование директорий
        self.clean_dir = self.data_dir / 'clean'
        self.noisy_dir = self.data_dir / 'noisy'
        self.mask_dir = self.data_dir / 'mask'
        
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")
        
        # Собираем список файлов
        self.clean_files = sorted(glob.glob(str(self.clean_dir / '*.png')) +
                                  glob.glob(str(self.clean_dir / '*.jpg')) +
                                  glob.glob(str(self.clean_dir / '*.bmp')))
        
        print(f"Found {len(self.clean_files)} clean images in {self.clean_dir}")
        
        # Проверяем наличие noisy и mask файлов
        self.has_noisy = self.noisy_dir.exists()
        self.has_mask = self.mask_dir.exists()
        
        if self.has_noisy:
            print(f"Using noisy images from {self.noisy_dir}")
        else:
            print("No noisy directory found. Will use clean images as noisy (for testing).")
        
        if self.has_mask:
            print(f"Using masks from {self.mask_dir}")
        elif synthetic_masks:
            print("No mask directory found. Will generate synthetic masks.")
        else:
            print("No mask directory found and synthetic_masks=False. Using empty masks.")
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Загрузка чистого изображения
        clean_path = self.clean_files[idx]
        clean_img = self.load_image(clean_path)
        
        # Получаем базовое имя файла
        filename = Path(clean_path).stem
        
        # Загрузка или создание зашумленного изображения
        if self.has_noisy:
            noisy_path = self.noisy_dir / (filename + Path(clean_path).suffix)
            if noisy_path.exists():
                noisy_img = self.load_image(str(noisy_path))
            else:
                # Если файл не найден, используем чистое изображение
                noisy_img = clean_img.copy()
                # Добавляем небольшой шум для разнообразия
                if self.mode == 'train':
                    noise = np.random.normal(0, 5, clean_img.shape).astype(np.float32)
                    noisy_img = np.clip(noisy_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            noisy_img = clean_img.copy()
        
        # Загрузка или создание маски
        if self.has_mask:
            mask_path = self.mask_dir / (filename + Path(clean_path).suffix)
            if mask_path.exists():
                mask_img = self.load_image(str(mask_path), is_mask=True)
            elif self.synthetic_masks:
                mask_img = self.generate_synthetic_mask(clean_img)
            else:
                mask_img = np.zeros_like(clean_img, dtype=np.uint8)
        elif self.synthetic_masks:
            mask_img = self.generate_synthetic_mask(clean_img)
        else:
            mask_img = np.zeros_like(clean_img, dtype=np.uint8)
        
        # Обрезка или паддинг до patch_size если нужно
        if self.patch_size > 0 and self.mode == 'train':
            clean_img, noisy_img, mask_img = self.random_crop(
                clean_img, noisy_img, mask_img, self.patch_size
            )
        
        # Конвертация в тензоры и нормализация
        clean_tensor = self.image_to_tensor(clean_img)
        noisy_tensor = self.image_to_tensor(noisy_img)
        mask_tensor = self.mask_to_tensor(mask_img)
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'mask': mask_tensor,
            'filename': filename
        }
    
    def load_image(self, path, is_mask=False):
        """Загрузка изображения"""
        try:
            # Пробуем разные методы загрузки
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Пробуем через PIL
                img = np.array(Image.open(path).convert('L'))
            
            if img is None:
                raise ValueError(f"Cannot load image: {path}")
            
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            return np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
    
    def generate_synthetic_mask(self, clean_img):
        """Генерация синтетической маски на основе чистого изображения"""
        # Простая бинаризация
        _, mask = cv2.threshold(clean_img, 127, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для улучшения маски
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def random_crop(self, clean, noisy, mask, crop_size):
        """Случайная обрезка изображений"""
        h, w = clean.shape
        
        if h < crop_size or w < crop_size:
            # Если изображение меньше crop_size, делаем паддинг
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            
            clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode='reflect')
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
            
            h, w = clean.shape
        
        # Случайные координаты для обрезки
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        clean_crop = clean[top:top+crop_size, left:left+crop_size]
        noisy_crop = noisy[top:top+crop_size, left:left+crop_size]
        mask_crop = mask[top:top+crop_size, left:left+crop_size]
        
        return clean_crop, noisy_crop, mask_crop
    
    def image_to_tensor(self, img):
        """Конвертация изображения в тензор с нормализацией"""
        # Конвертация в float и нормализация в диапазон [-1, 1]
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor * 2 - 1  # [-1, 1]
        img_tensor = img_tensor.unsqueeze(0)  # Добавляем канальный размер
        return img_tensor
    
    def mask_to_tensor(self, mask):
        """Конвертация маски в тензор"""
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0)  # Добавляем канальный размер
        return mask_tensor


class SyntheticNoiseMaskDataset(Dataset):
    """
    Синтетический датасет с генерируемыми на лету данными
    """
    def __init__(self, num_samples=1000, patch_size=128, 
                 noise_levels=[15, 25, 50], transform=None):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_levels = noise_levels
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Генерация чистого изображения
        clean_img = self.generate_clean_image()
        
        # Генерация маски
        mask_img = self.generate_mask(clean_img)
        
        # Добавление шума
        noise_level = random.choice(self.noise_levels)
        noisy_img = self.add_noise(clean_img, noise_level)
        
        # Конвертация в тензоры
        clean_tensor = self.image_to_tensor(clean_img)
        noisy_tensor = self.image_to_tensor(noisy_img)
        mask_tensor = self.mask_to_tensor(mask_img)
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'mask': mask_tensor,
            'filename': f'synthetic_{idx:06d}'
        }
    
    def generate_clean_image(self):
        """Генерация чистого изображения с геометрическими фигурами"""
        img = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Случайное количество фигур
        num_shapes = random.randint(3, 8)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['line', 'circle', 'rectangle', 'polygon'])
            
            if shape_type == 'line':
                pt1 = (random.randint(0, self.patch_size), 
                       random.randint(0, self.patch_size))
                pt2 = (random.randint(0, self.patch_size), 
                       random.randint(0, self.patch_size))
                thickness = random.randint(1, 3)
                cv2.line(img, pt1, pt2, 255, thickness)
            
            elif shape_type == 'circle':
                radius = random.randint(5, 20)
                center = (random.randint(radius, self.patch_size - radius),
                         random.randint(radius, self.patch_size - radius))
                thickness = random.choice([-1, 1, 2])
                cv2.circle(img, center, radius, 255, thickness)
            
            elif shape_type == 'rectangle':
                pt1 = (random.randint(0, self.patch_size - 40),
                      random.randint(0, self.patch_size - 40))
                pt2 = (pt1[0] + random.randint(20, 40),
                      pt1[1] + random.randint(20, 40))
                thickness = random.choice([-1, 1, 2])
                cv2.rectangle(img, pt1, pt2, 255, thickness)
            
            elif shape_type == 'polygon':
                num_vertices = random.randint(3, 6)
                vertices = []
                for _ in range(num_vertices):
                    vertices.append([random.randint(0, self.patch_size), 
                                   random.randint(0, self.patch_size)])
                pts = np.array([vertices], dtype=np.int32)
                thickness = random.choice([-1, 1])
                cv2.fillPoly(img, pts, 255)
        
        return img
    
    def generate_mask(self, clean_img):
        """Генерация маски на основе чистого изображения"""
        # Простая бинаризация
        _, mask = cv2.threshold(clean_img, 127, 255, cv2.THRESH_BINARY)
        return mask
    
    def add_noise(self, image, noise_level):
        """Добавление гауссовского шума"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def image_to_tensor(self, img):
        """Конвертация изображения в тензор"""
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_tensor = img_tensor * 2 - 1  # [-1, 1]
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def mask_to_tensor(self, mask):
        """Конвертация маски в тензор"""
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor