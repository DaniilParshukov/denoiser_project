# denoiser_project/vectorize.py
import cv2
import numpy as np
import potrace
import svgwrite
from pathlib import Path
import argparse
import json
from xml.dom import minidom


class MaskVectorizer:
    """Класс для векторизации бинарных масок в SVG"""
    
    def __init__(self, simplify_tolerance=0.1, min_area=10, 
                 corner_threshold=100.0, optimize_paths=True):
        """
        Args:
            simplify_tolerance: допуск для упрощения контуров
            min_area: минимальная площадь контура для сохранения
            corner_threshold: порог для определения углов
            optimize_paths: оптимизировать SVG пути
        """
        self.simplify_tolerance = simplify_tolerance
        self.min_area = min_area
        self.corner_threshold = corner_threshold
        self.optimize_paths = optimize_paths
        
    def preprocess_mask(self, mask):
        """
        Предобработка маски перед векторизацией
        """
        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
        # Бинаризация если нужно
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Пороговая обработка
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для очистки
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def mask_to_contours(self, mask):
        """
        Извлечение контуров из маски с помощью OpenCV
        """
        # Находим контуры
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Фильтруем маленькие контуры
        filtered_contours = []
        filtered_hierarchy = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if area >= self.min_area:
                    # Упрощение контура
                    epsilon = self.simplify_tolerance * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 3:  # Нужно минимум 3 точки для полигона
                        filtered_contours.append(approx)
                        filtered_hierarchy.append(hierarchy[i])
        
        return filtered_contours, filtered_hierarchy
    
    def potrace_vectorize(self, mask):
        """
        Векторизация с помощью Potrace
        """
        # Подготавливаем маску для Potrace
        # Potrace ожидает белый на черном фоне
        bitmap_data = (mask == 255).astype(np.uint8)
        
        # Создаем Bitmap для Potrace
        bitmap = potrace.Bitmap(bitmap_data)
        
        # Трассировка
        path = bitmap.trace(
            turdsize=2,
            turnpolicy=potrace.TURNPOLICY_MINORITY,
            alphamax=1.0,
            opticurve=True,
            opttolerance=0.2
        )
        
        return path
    
    def contours_to_svg_paths(self, contours, hierarchy=None):
        """
        Конвертация контуров OpenCV в SVG пути
        """
        svg_paths = []
        
        if hierarchy is not None:
            # Используем иерархию для правильного определения отверстий
            for i, contour in enumerate(contours):
                # Проверяем, является ли это отверстием
                is_hole = hierarchy[i][3] >= 0  # parent exists
                
                # Создаем путь
                points = contour.squeeze()
                if len(points) < 2:
                    continue
                
                # Начинаем путь с первой точки
                path_data = f"M {points[0][0]},{points[0][1]}"
                
                # Добавляем остальные точки
                for point in points[1:]:
                    path_data += f" L {point[0]},{point[1]}"
                
                # Замыкаем путь
                path_data += " Z"
                
                svg_paths.append({
                    'd': path_data,
                    'is_hole': is_hole,
                    'area': cv2.contourArea(contour)
                })
        else:
            # Без иерархии
            for contour in contours:
                points = contour.squeeze()
                if len(points) < 2:
                    continue
                
                path_data = f"M {points[0][0]},{points[0][1]}"
                for point in points[1:]:
                    path_data += f" L {point[0]},{point[1]}"
                path_data += " Z"
                
                svg_paths.append({
                    'd': path_data,
                    'is_hole': False,
                    'area': cv2.contourArea(contour)
                })
        
        return svg_paths
    
    def potrace_to_svg_paths(self, potrace_path):
        """
        Конвертация результата Potrace в SVG пути
        """
        svg_paths = []
        
        for curve in potrace_path:
            path_data = []
            start = curve.start_point
            
            # Начинаем путь
            path_data.append(f"M {start.x},{start.y}")
            
            for segment in curve:
                if segment.is_corner:
                    # Угловой сегмент: две прямые линии
                    path_data.append(f"L {segment.c.x},{segment.c.y}")
                    path_data.append(f"L {segment.end_point.x},{segment.end_point.y}")
                else:
                    # Кривая Безье
                    path_data.append(
                        f"C {segment.c1.x},{segment.c1.y} "
                        f"{segment.c2.x},{segment.c2.y} "
                        f"{segment.end_point.x},{segment.end_point.y}"
                    )
            
            # Замыкаем путь если нужно
            if curve.is_closed:
                path_data.append("Z")
            
            svg_paths.append({
                'd': " ".join(path_data),
                'is_hole': False,  # Potrace не дает информацию о отверстиях
                'area': 0  # Нужно вычислять отдельно
            })
        
        return svg_paths
    
    def optimize_svg_paths(self, svg_paths):
        """
        Оптимизация SVG путей
        """
        if not self.optimize_paths:
            return svg_paths
        
        optimized_paths = []
        
        for path_info in svg_paths:
            path_data = path_info['d']
            
            # Упрощение: удаление повторяющихся команд
            # Замена "M x,y L x,y" на "M x,y"
            import re
            path_data = re.sub(r'M ([\d.]+),([\d.]+) L \1,\2', r'M \1,\2', path_data)
            
            # Удаление лишних пробелов
            path_data = ' '.join(path_data.split())
            
            path_info['d'] = path_data
            optimized_paths.append(path_info)
        
        return optimized_paths
    
    def create_svg_document(self, svg_paths, width, height, 
                           stroke_color='black', stroke_width=1,
                           fill_color='black', background_color='white'):
        """
        Создание SVG документа из путей
        """
        # Создаем документ
        dwg = svgwrite.Drawing(size=(width, height))
        
        # Фон
        if background_color:
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(width, height),
                fill=background_color
            ))
        
        # Добавляем пути
        for path_info in svg_paths:
            path = dwg.path(
                d=path_info['d'],
                fill=fill_color if not path_info['is_hole'] else background_color,
                stroke=stroke_color,
                stroke_width=stroke_width
            )
            dwg.add(path)
        
        return dwg
    
    def vectorize(self, mask, method='opencv'):
        """
        Основная функция векторизации
        """
        # Предобработка
        processed_mask = self.preprocess_mask(mask)
        height, width = processed_mask.shape
        
        if method == 'opencv':
            # Метод OpenCV
            contours, hierarchy = self.mask_to_contours(processed_mask)
            svg_paths = self.contours_to_svg_paths(contours, hierarchy)
        
        elif method == 'potrace':
            # Метод Potrace
            potrace_path = self.potrace_vectorize(processed_mask)
            svg_paths = self.potrace_to_svg_paths(potrace_path)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Оптимизация
        svg_paths = self.optimize_svg_paths(svg_paths)
        
        # Создание SVG
        svg_doc = self.create_svg_document(
            svg_paths, width, height,
            stroke_color='black',
            stroke_width=1,
            fill_color='black',
            background_color='white'
        )
        
        # Статистика
        stats = {
            'width': width,
            'height': height,
            'num_paths': len(svg_paths),
            'num_points': sum(len(p['d'].split()) for p in svg_paths),
            'total_area': sum(p['area'] for p in svg_paths),
            'method': method
        }
        
        return svg_doc, stats, svg_paths
    
    def save_svg(self, svg_doc, output_path, prettify=True):
        """
        Сохранение SVG документа
        """
        svg_string = svg_doc.tostring()
        
        if prettify:
            # Форматирование XML
            xml_dom = minidom.parseString(svg_string)
            pretty_xml = xml_dom.toprettyxml(indent='  ')
            
            # Удаление лишних пустых строк
            lines = pretty_xml.split('\n')
            lines = [line for line in lines if line.strip()]
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
        else:
            with open(output_path, 'w') as f:
                f.write(svg_string)
    
    def vectorize_folder(self, input_dir, output_dir, method='opencv',
                        image_extensions=('.png', '.jpg', '.jpeg', '.bmp')):
        """
        Векторизация всех масок в папке
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поиск файлов
        mask_files = []
        for ext in image_extensions:
            mask_files.extend(input_path.glob(f'*{ext}'))
            mask_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(mask_files)} mask files in {input_dir}")
        
        results = []
        for mask_file in mask_files:
            try:
                print(f"Vectorizing {mask_file.name}...")
                
                # Векторизация
                svg_doc, stats, paths = self.vectorize(str(mask_file), method)
                
                # Сохранение
                output_svg = output_path / f"{mask_file.stem}.svg"
                self.save_svg(svg_doc, str(output_svg))
                
                # Сохранение статистики
                stats_json = output_path / f"{mask_file.stem}_stats.json"
                with open(stats_json, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # Сохранение путей в отдельный файл
                paths_json = output_path / f"{mask_file.stem}_paths.json"
                with open(paths_json, 'w') as f:
                    json.dump(paths, f, indent=2, default=str)
                
                results.append({
                    'filename': mask_file.name,
                    'svg_path': str(output_svg),
                    'stats': stats
                })
                
                print(f"  ✓ Created SVG with {stats['num_paths']} paths")
                
            except Exception as e:
                print(f"  ✗ Error vectorizing {mask_file.name}: {e}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Vectorize binary masks to SVG')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input mask image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--method', type=str, default='opencv',
                       choices=['opencv', 'potrace'],
                       help='Vectorization method')
    parser.add_argument('--simplify', type=float, default=0.1,
                       help='Contour simplification tolerance')
    parser.add_argument('--min_area', type=int, default=10,
                       help='Minimum contour area')
    parser.add_argument('--no_optimize', action='store_true',
                       help='Disable path optimization')
    parser.add_argument('--batch', action='store_true',
                       help='Process entire directory')
    
    args = parser.parse_args()
    
    # Создание векторизатора
    vectorizer = MaskVectorizer(
        simplify_tolerance=args.simplify,
        min_area=args.min_area,
        optimize_paths=not args.no_optimize
    )
    
    # Проверка: файл или папка
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Обработка папки
        print(f"Processing directory: {args.input}")
        results = vectorizer.vectorize_folder(
            args.input, args.output, args.method
        )
        
        print(f"\nVectorized {len(results)} masks")
        print(f"Results saved to {args.output}")
        
        # Сводная статистика
        if results:
            total_paths = sum(r['stats']['num_paths'] for r in results)
            avg_paths = total_paths / len(results)
            print(f"Average paths per mask: {avg_paths:.1f}")
    
    elif input_path.is_file():
        # Обработка одного файла
        print(f"Vectorizing single mask: {args.input}")
        
        svg_doc, stats, paths = vectorizer.vectorize(
            args.input, args.method
        )
        
        # Сохранение
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stem = input_path.stem
        svg_file = output_path / f"{stem}.svg"
        vectorizer.save_svg(svg_doc, str(svg_file))
        
        # Статистика
        stats_file = output_path / f"{stem}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nCreated SVG: {svg_file}")
        print(f"Size: {stats['width']}x{stats['height']}")
        print(f"Paths: {stats['num_paths']}")
        print(f"Total area: {stats['total_area']:.0f} pixels")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()