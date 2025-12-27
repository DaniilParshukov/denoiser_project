import torch
import torch.nn as nn
import torch.nn.init as init

class DnCNNWithMask(nn.Module):
    """
    DnCNN архитектура с двумя выходами:
    1. Денойзинг (очищенное изображение)
    2. Маска (бинарная маска для векторизации)
    """
    
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True):
        super(DnCNNWithMask, self).__init__()
        
        kernel_size = 3
        padding = 1
        
        # Общие слои (энкодер)
        self.shared_layers = nn.ModuleList()
        
        # Первый слой
        self.shared_layers.append(nn.Conv2d(
            image_channels, n_channels, kernel_size, padding=padding, bias=True
        ))
        self.shared_layers.append(nn.ReLU(inplace=True))
        
        # Скрытые слои
        for _ in range(depth - 2):
            self.shared_layers.append(nn.Conv2d(
                n_channels, n_channels, kernel_size, padding=padding, bias=False
            ))
            
            if use_bnorm:
                self.shared_layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            
            self.shared_layers.append(nn.ReLU(inplace=True))
        
        # Два отдельных выхода
        # 1. Для денойзинга (предсказывает шум)
        self.denoise_output = nn.Conv2d(
            n_channels, image_channels, kernel_size, padding=padding, bias=False
        )
        
        # 2. Для маски (предсказывает вероятность пикселя)
        self.mask_output = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 2, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()  # Выход [0, 1] - вероятность
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Пропускаем через общие слои
        features = x
        for layer in self.shared_layers:
            features = layer(features)
        
        # Получаем два выхода
        noise = self.denoise_output(features)
        denoised = x - noise  # Residual learning для денойзинга
        
        mask = self.mask_output(features)
        
        return denoised, mask


class SimpleDnCNNMask(nn.Module):
    """
    Упрощенная версия для быстрого прототипирования
    """
    def __init__(self, channels=1):
        super(SimpleDnCNNMask, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Декодер для денойзинга
        self.denoise_decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1),
        )
        
        # Декодер для маски
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        noise = self.denoise_decoder(features)
        denoised = x - noise
        
        mask = self.mask_decoder(features)
        
        return denoised, mask


if __name__ == "__main__":
    # Тестирование модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DnCNNWithMask(depth=7, n_channels=32).to(device)
    test_input = torch.randn(2, 1, 256, 256).to(device)
    
    with torch.no_grad():
        denoised, mask = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Denoised shape: {denoised.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")