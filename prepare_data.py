#!/usr/bin/env python3
"""
Скрипт для подготовки данных для обучения
Генерирует синтетические данные или подготавливает существующие
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
import shutil
from tqdm import tqdm


class DataPreparer:
    """Класс для подготовки данных обучения"""
    
    @staticmethod
    def create_synthetic_dataset(output_dir, num_images=1000, 
                                image_size=256, noise_levels=[15, 25, 50]):
        """
        Создание синтетического датасета с чистыми изображениями и масками
        """
        # Создание структуры папок
        train_dir = Path(output_dir) / 'train'
        test_dir = Path(output_dir) / 'test'
        
        for mode_dir in [train_dir, test_dir]:
            (mode_dir / 'clean').mkdir(parents=True, exist_ok=True)
            (mode_dir / 'noisy').mkdir(parents=True, exist_ok=True)
            (mode_dir / 'mask').mkdir(parents=True, exist_ok=True)
        
        # Разделение на train/test
        train_count = int(num_images * 0.8)
        test_count = num_images - train_count
        
        print(f"Creating synthetic dataset with {num_images} images")
        print(f"  - Train: {train_count} images")
        print(f"  - Test: {test_count} images")
        print(f"  - Size: {image_size}x{image_size}")
        
        # Генерация изображений
        for i in tqdm(range(num_images)):
            # Определяем режим (train/test)
            if i < train_count:
                mode = 'train'
            else:
                mode = 'test'
            
            # Генерация чистого изображения с фигурами
            clean = DataPreparer.generate_clean_image(image_size)
            
            # Генерация маски (просто бинаризация чистого)
            mask = (clean > 127).astype(np.uint8) * 255
            
            # Добавление шума
            noise_level = random.choice(noise_levels)
            noisy = DataPreparer.add_noise(clean, noise_level)
            
            # Сохранение
            idx = i if mode == 'train' else i - train_count
            base_name = f"{mode}_{idx:04d}"
            
            mode_path = train_dir if mode == 'train' else test_dir
            
            cv2.imwrite(str(mode_path / 'clean' / f'{base_name}.png'), clean)
            cv2.imwrite(str(mode_path / 'mask' / f'{base_name}.png'), mask)
            cv2.imwrite(str(mode_path / 'noisy' / f'{base_name}.png'), noisy)
        
        print(f"\nDataset created in {output_dir}")
        
        # Создание файла с информацией о датасете
        info = {
            'total_images': num_images,
            'train_count': train_count,
            'test_count': test_count,
            'image_size': image_size,
            'noise_levels': noise_levels,
            'generated_at': str(np.datetime64('now'))
        }
        
        import json
        with open(Path(output_dir) / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    @staticmethod
    def generate_clean_image(size):
        """Генерация чистого изображения с геометрическими фигурами"""
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Случайное количество фигур
        num_shapes = random.randint(3, 8)
        
        for _ in range(num_shapes):
            shape_type = random.choice(['line', 'circle', 'rectangle', 'polygon'])
            
            if shape_type == 'line':
                # Линия
                pt1 = (random.randint(0, size), random.randint(0, size))
                pt2 = (random.randint(0, size), random.randint(0, size))
                thickness = random.randint(1, 3)
                cv2.line(img, pt1, pt2, 255, thickness)
            
            elif shape_type == 'circle':
                # Круг
                radius = random.randint(5, 20)
                center = (random.randint(radius, size - radius),
                         random.randint(radius, size - radius))
                thickness = random.choice([-1, 1, 2])
                cv2.circle(img, center, radius, 255, thickness)
            
            elif shape_type == 'rectangle':
                # Прямоугольник
                pt1 = (random.randint(0, size - 40),
                      random.randint(0, size - 40))
                pt2 = (pt1[0] + random.randint(20, 40),
                      pt1[1] + random.randint(20, 40))
                thickness = random.choice([-1, 1, 2])
                cv2.rectangle(img, pt1, pt2, 255, thickness)
            
            elif shape_type == 'polygon':
                # Многоугольник
                num_vertices = random.randint(3, 6)
                vertices = []
                for _ in range(num_vertices):
                    vertices.append([random.randint(0, size), 
                                   random.randint(0, size)])
                pts = np.array([vertices], dtype=np.int32)
                thickness = random.choice([-1, 1])
                cv2.fillPoly(img, pts, 255)
        
        return img
    
    @staticmethod
    def add_noise(image, noise_level):
        """Добавление гауссовского шума"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def prepare_existing_dataset(input_dir, output_dir, 
                               train_ratio=0.8, add_synthetic_noise=True,
                               noise_levels=[15, 25, 50]):
        """
        Подготовка существующего датасета
        Предполагается, что в input_dir есть чистые изображения
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Получение списка изображений
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        clean_files = []
        
        for ext in image_extensions:
            clean_files.extend(input_path.glob(f'*{ext}'))
            clean_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(clean_files)} clean images in {input_dir}")
        
        if len(clean_files) == 0:
            print("No images found. Creating synthetic dataset instead.")
            DataPreparer.create_synthetic_dataset(output_dir, 1000)
            return
        
        # Перемешивание
        random.shuffle(clean_files)
        
        # Разделение на train/test
        split_idx = int(len(clean_files) * train_ratio)
        train_files = clean_files[:split_idx]
        test_files = clean_files[split_idx:]
        
        print(f"  - Train: {len(train_files)} images")
        print(f"  - Test: {len(test_files)} images")
        
        # Создание структуры папок
        train_dirs = ['clean', 'noisy', 'mask']
        test_dirs = ['clean', 'noisy', 'mask']
        
        for mode, files in [('train', train_files), ('test', test_files)]:
            mode_dir = output_path / mode
            for dir_name in (train_dirs if mode == 'train' else test_dirs):
                (mode_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Обработка train изображений
        print("\nProcessing train images...")
        for i, img_path in enumerate(tqdm(train_files)):
            clean = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if clean is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Масштабирование если нужно
            if clean.shape[0] > 512 or clean.shape[1] > 512:
                scale = 512 / max(clean.shape)
                new_size = (int(clean.shape[1] * scale), 
                           int(clean.shape[0] * scale))
                clean = cv2.resize(clean, new_size)
            
            # Создание маски (простая бинаризация)
            _, mask = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
            
            # Добавление шума если нужно
            if add_synthetic_noise:
                noise_level = random.choice(noise_levels)
                noisy = DataPreparer.add_noise(clean, noise_level)
            else:
                # Если нет шума, копируем чистое изображение
                noisy = clean.copy()
            
            # Сохранение
            base_name = f"train_{i:04d}"
            cv2.imwrite(str(output_path / 'train' / 'clean' / f'{base_name}.png'), clean)
            cv2.imwrite(str(output_path / 'train' / 'mask' / f'{base_name}.png'), mask)
            cv2.imwrite(str(output_path / 'train' / 'noisy' / f'{base_name}.png'), noisy)
        
        # Обработка test изображений
        print("\nProcessing test images...")
        for i, img_path in enumerate(tqdm(test_files)):
            clean = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if clean is None:
                continue
            
            # Масштабирование если нужно
            if clean.shape[0] > 512 or clean.shape[1] > 512:
                scale = 512 / max(clean.shape)
                new_size = (int(clean.shape[1] * scale), 
                           int(clean.shape[0] * scale))
                clean = cv2.resize(clean, new_size)
            
            # Создание маски
            _, mask = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
            
            # Для теста добавляем только немного шума или оставляем чистым
            if add_synthetic_noise:
                noise_level = noise_levels[0]  # Минимальный уровень
                noisy = DataPreparer.add_noise(clean, noise_level)
            else:
                noisy = clean.copy()
            
            # Сохранение
            base_name = f"test_{i:04d}"
            cv2.imwrite(str(output_path / 'test' / 'clean' / f'{base_name}.png'), clean)
            cv2.imwrite(str(output_path / 'test' / 'mask' / f'{base_name}.png'), mask)
            cv2.imwrite(str(output_path / 'test' / 'noisy' / f'{base_name}.png'), noisy)
        
        print(f"\nDataset prepared in {output_dir}")
        
        # Информация о датасете
        info = {
            'source_dir': input_dir,
            'total_images': len(clean_files),
            'train_count': len(train_files),
            'test_count': len(test_files),
            'add_synthetic_noise': add_synthetic_noise,
            'noise_levels': noise_levels if add_synthetic_noise else [],
            'prepared_at': str(np.datetime64('now'))
        }
        
        import json
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Команда для создания синтетического датасета
    synth_parser = subparsers.add_parser('synthetic', 
                                         help='Create synthetic dataset')
    synth_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    synth_parser.add_argument('--num_images', type=int, default=1000,
                             help='Number of images to generate')
    synth_parser.add_argument('--image_size', type=int, default=256,
                             help='Image size')
    synth_parser.add_argument('--noise_levels', nargs='+', type=int,
                             default=[15, 25, 50],
                             help='Noise levels')
    
    # Команда для подготовки существующего датасета
    prep_parser = subparsers.add_parser('prepare', 
                                        help='Prepare existing dataset')
    prep_parser.add_argument('--input', type=str, required=True,
                            help='Input directory with clean images')
    prep_parser.add_argument('--output', type=str, required=True,
                            help='Output directory')
    prep_parser.add_argument('--train_ratio', type=float, default=0.8,
                            help='Train/test split ratio')
    prep_parser.add_argument('--no_synthetic_noise', action='store_true',
                            help='Do not add synthetic noise')
    prep_parser.add_argument('--noise_levels', nargs='+', type=int,
                            default=[15, 25, 50],
                            help='Noise levels for synthetic noise')
    
    args = parser.parse_args()
    
    if args.command == 'synthetic':
        DataPreparer.create_synthetic_dataset(
            args.output, args.num_images, 
            args.image_size, args.noise_levels
        )
    
    elif args.command == 'prepare':
        DataPreparer.prepare_existing_dataset(
            args.input, args.output,
            args.train_ratio,
            not args.no_synthetic_noise,
            args.noise_levels
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()