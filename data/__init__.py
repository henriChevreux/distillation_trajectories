"""
Data module for CTM project.
Contains dataset loading and preprocessing utilities.
"""

from .dataset import get_data_loader, get_dataset_stats

__all__ = [
    'get_data_loader',
    'get_dataset_stats',
] 