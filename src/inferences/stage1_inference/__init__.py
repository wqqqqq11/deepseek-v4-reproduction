"""
阶段1推理模块
"""

from .generator import Generator
from .sampler import sample_next_token

__all__ = ["Generator", "sample_next_token"]
