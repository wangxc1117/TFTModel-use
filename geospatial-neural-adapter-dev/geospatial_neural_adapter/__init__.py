"""
Geospatial Neural Adapter

A Python package for neural spatial modeling with low-rank approximations.
"""

__version__ = "0.6.0"
__author__ = "Wen-Ting Wang"
__email__ = "egpivo@gmail.com"

# Import data generation functions
from .data.generators import (
    generate_combined_synthetic_data,
    generate_time_synthetic_data,
)
from .data.preprocessing import prepare_all

# Import metrics
from .metrics import compute_metrics, frobenius_norm, fusion_score

# Import main classes for easier access
from .models.spatial_basis_learner import SpatialBasisLearner
from .models.spatial_neural_adapter import SpatialNeuralAdapter
from .models.trend_model import TrendModel

__all__ = [
    "SpatialBasisLearner",
    "TrendModel",
    "SpatialNeuralAdapter",
    "generate_combined_synthetic_data",
    "generate_time_synthetic_data",
    "prepare_all",
    "fusion_score",
    "frobenius_norm",
    "compute_metrics",
]
