from importlib.metadata import PackageNotFoundError, version

from ._configurators_v2 import KernelsConfigV2, init_to_zero_v2
from .flash_attention_2 import flash_attention_v2, flash_attention_v2_custom

try:
    __version__ = version("flash-attention-triton")
except PackageNotFoundError:
    __version__ = "0.0.dev0"  # fallback for development

__author__ = "Egor Zakharenko"
__email__ = "egorzakharenko97@gmail.com"
__all__ = ["KernelsConfigV2", "flash_attention_v2", "flash_attention_v2_custom", "init_to_zero_v2"]
