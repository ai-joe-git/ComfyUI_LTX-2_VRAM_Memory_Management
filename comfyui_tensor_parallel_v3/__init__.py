"""
Tensor Parallel V3 - Safe FFN Chunking
======================================

This version ONLY does FFN chunking - no multi-GPU operations.
Guaranteed safe with ComfyUI's offloading mode.

For multi-GPU attention parallelism, disable offloading and use V2.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorParallelV3")

__version__ = "3.0.0"

def _init():
    import torch
    logger.info("=" * 60)
    logger.info("Tensor Parallel V3 - Safe FFN Chunking")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This version is safe with ALL modes:")
    logger.info("  - Works with offloading (checkmark)")
    logger.info("  - Works with lowvram (checkmark)")
    logger.info("  - No multi-GPU conflicts (checkmark)")
    logger.info("")
    logger.info("For multi-GPU attention, use V2 with offloading disabled.")
    logger.info("=" * 60)

_init()

from .tensor_parallel_v3 import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
