# denoiser_project/models/__init__.py
from .dncnn import DnCNNWithMask, SimpleDnCNNMask

__all__ = ['DnCNNWithMask', 'SimpleDnCNNMask']