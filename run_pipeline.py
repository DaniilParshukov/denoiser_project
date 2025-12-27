#!/usr/bin/env python3
"""
Полный пайплайн: денойзинг → маска → векторизация
"""

import os
import argparse
from pathlib import Path
import sys

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from denoise import ImageDenoiserWithMask
from vectorize import MaskVectorizer


class CompletePipeline:
    """Полный пайплайн обработки изображений"""
    
    def __init__(self, model_path, denoise_threshold=0.5, 
                 vectorize_method='opencv', simplify_tolerance=0.1):
        """
        Args:
            model_path: путь к модели денойзинга
            denoise_threshold: порог для бинаризации маски
            vectorize_method: метод векторизации ('opencv' или 'potrace')
            simplify_tolerance: допуск упрощения контуров
        """
        self.denoise_threshold = denoise_threshold
        self.vectorize_method = vectorize_method
        
        # Инициализация компонентов
        print("Initializing pipeline components...")
        self.denoiser = ImageDenoiserWithMask(model_path)
        self.vectorizer = MaskVectorizer(
            simplify_tolerance=simplify_tolerance,
            optimize_paths=True
        )
    
    def process_single_image(self, input_image, output_dir):
        """
        Обработка одного изображения
        """
        input_path = Path(input_image)
        output_path = Path(output_dir)
        
        # Создание подпапок
        intermediate_dir = output_path / 'intermediate'
        final_dir = output_path / 'final'
        
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        stem = input_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_path.name}")
        print(f"{'='*60}")
        
        # Шаг 1: Денойзинг и получение маски
        print("\nStep 1: Denoising and mask extraction...")
        denoise_results = self.denoiser.denoise_image(
            str(input_path), self.denoise_threshold
        )
        
        # Сохранение промежуточных результатов
        mask_path = intermediate_dir / f"{stem}_mask_binary.png"
        cv2.imwrite(str(mask_path), denoise_results['mask_binary'])
        
        print(f"  ✓ Mask saved: {mask_path}")
        
        # Шаг 2: Векторизация маски
        print("\nStep 2: Vectorizing mask...")
        svg_doc, stats, paths = self.vectorizer.vectorize(
            str(mask_path), self.vectorize_method
        )
        
        # Сохранение SVG
        svg_path = final_dir / f"{stem}.svg"
        self.vectorizer.save_svg(svg_doc, str(svg_path))
        
        # Сохранение статистики
        stats_path = final_dir / f"{stem}_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nResults:")
        print(f"  ✓ Original: {input_path}")
        print(f"  ✓ Mask: {mask_path}")
        print(f"  ✓ SVG: {svg_path}")
        print(f"  ✓ Stats: {stats_path}")
        print(f"\nStatistics:")
        print(f"  - Image size: {stats['width']}x{stats['height']}")
        print(f"  - Number of paths: {stats['num_paths']}")
        print(f"  - Total area: {stats['total_area']:.0f} pixels")
        
        return {
            'input': str(input_path),
            'mask': str(mask_path),
            'svg': str(svg_path),
            'stats': stats,
            'denoised': denoise_results['denoised'],
            'mask_binary': denoise_results['mask_binary']
        }
    
    def process_batch(self, input_dir, output_dir, 
                     image_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        """
        Пакетная обработка изображений
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Создание структуры папок
        intermediate_dir = output_path / 'intermediate'
        final_dir = output_path / 'final'
        logs_dir = output_path / 'logs'
        
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Поиск изображений
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        print(f"Output will be saved to {output_dir}")
        
        results = []
        failed = []
        
        for i, img_path in enumerate(image_files):
            try:
                print(f"\n[{i+1}/{len(image_files)}] Processing {img_path.name}...")
                
                result = self.process_single_image(str(img_path), output_path)
                results.append(result)
                
                # Прогресс
                print(f"  ✓ Completed")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed.append(str(img_path))
        
        # Сводный отчет
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print("\nFailed files:")
            for f in failed:
                print(f"  - {f}")
        
        # Сохранение сводного отчета
        if results:
            summary = {
                'total_processed': len(results),
                'failed': failed,
                'summary_stats': {
                    'avg_paths': sum(r['stats']['num_paths'] for r in results) / len(results),
                    'avg_area': sum(r['stats']['total_area'] for r in results) / len(results),
                }
            }
            
            import json
            summary_path = logs_dir / 'batch_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_path}")
        
        return results, failed


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Denoise → Extract Mask → Vectorize to SVG'
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to denoising model')
    parser.add_argument('--denoise_threshold', type=float, default=0.5,
                       help='Threshold for mask binarization')
    parser.add_argument('--vectorize_method', type=str, default='opencv',
                       choices=['opencv', 'potrace'],
                       help='Vectorization method')
    parser.add_argument('--simplify', type=float, default=0.1,
                       help='Contour simplification tolerance')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in directory')
    
    args = parser.parse_args()
    
    # Инициализация пайплайна
    pipeline = CompletePipeline(
        model_path=args.model,
        denoise_threshold=args.denoise_threshold,
        vectorize_method=args.vectorize_method,
        simplify_tolerance=args.simplify
    )
    
    # Запуск обработки
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        print("Starting batch processing...")
        results, failed = pipeline.process_batch(
            args.input, args.output
        )
    elif input_path.is_file():
        print("Processing single image...")
        result = pipeline.process_single_image(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    # Нужно импортировать cv2 здесь чтобы избежать циклических импортов
    import cv2
    main()