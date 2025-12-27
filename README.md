## Установка
```bash
pip install -r requirements.txt
```
## 1. Подготовка данных
# Создание синтетического датасета
```bash
python prepare_data.py synthetic --output data --num_images 1000
```

# Или подготовка существующего датасета
```bash
python prepare_data.py prepare --input path/to/clean/images --output data
```

## 2. Обучение модели
# Обучение на синтетических данных
```bash
python train.py --synthetic --epochs 50 --batch_size 16
```

# Обучение на реальных данных
```bash
python train.py --data_dir data --epochs 50 --batch_size 16
```

## 3. Денойзинг изображений
# Обработка одного изображения
```bash
python denoise.py --input noisy_image.jpg --output results --model checkpoints/best_model.pth
```

# Обработка папки
```bash
python denoise.py --input noisy_images/ --output results --model checkpoints/best_model.pth --save_all
```

## 4. Векторизация масок
# Векторизация одной маски
```bash
python vectorize.py --input mask.png --output svg_results --method opencv
```

# Пакетная векторизация
```bash
python vectorize.py --input masks/ --output svg_results --method potrace --batch
```

## 5. Полный пайплайн
# Одно изображение
```bash
python run_pipeline.py --input noisy_image.jpg --output final_results
```

# Пакетная обработка
```bash
python run_pipeline.py --input noisy_images/ --output batch_results --batch
```

## Обучение
```bash
python train.py \
  --model simple \          # Архитектура модели (simple или dncnn)
  --depth 7 \               # Количество слоёв
  --epochs 50 \             # Количество эпох
  --batch_size 16 \         # Размер батча
  --lr 0.001 \              # Скорость обучения
  --denoise_weight 1.0 \    # Вес для loss денойзинга
  --mask_weight 1.0 \       # Вес для loss маски
  --mask_loss bce_dice \    # Функция потерь для маски
  --synthetic \             # Использовать синтетические данные
  --patch_size 128          # Размер патча
```

Векторизация
```bash
python vectorize.py \
  --method opencv \        # Метод векторизации (opencv или potrace)
  --simplify 0.1 \         # Допуск упрощения контуров
  --min_area 10 \          # Минимальная площадь контура
  --no_optimize            # Отключить оптимизацию путей
```

# Архитектура DnCNN из оригинальной статьи
https://arxiv.org/abs/1608.03981

# Структура
denoiser_project/
├── data/               # Данные для обучения и тестирования
│ ├── train/            # Обучающая выборка
│ │ ├── noisy/          # Зашумленные изображения
│ │ ├── clean/          # Чистые изображения
│ │ └── mask/           # Бинарные маски
│ └── test/             # Тестовая выборка
│   ├── noisy/
│   ├── clean/
│   └── mask/
├── models/             # Модели нейросетей
│ ├── dncnn.py          # Архитектура DnCNN с маской
│ └── pretrained/       # Предобученные модели
├── utils/              # Вспомогательные модули
│ ├── dataset.py        # Датасеты и аугментации
│ └── metrics.py        # Метрики качества
├── checkpoints/        # Сохранённые веса моделей
├── logs/               # Логи TensorBoard
├── visualizations/     # Визуализации результатов
├── train.py            # Обучение модели
├── denoise.py          # Денойзинг изображений
├── vectorize.py        # Векторизация масок в SVG
├── run_pipeline.py     # Полный пайплайн
├── prepare_data.py     # Подготовка данных
└── requirements.txt    # Зависимости Python