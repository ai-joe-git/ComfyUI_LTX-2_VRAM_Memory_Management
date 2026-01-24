"""
Tensor Parallel V3 - Safe Chunking Mode
=======================================

This version ONLY does FFN chunking - no multi-GPU operations.
Guaranteed compatible with ComfyUI's offloading mode.

For multi-GPU attention parallelism, use V2 with offloading DISABLED.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorParallelV3")


class ChunkedFFN(nn.Module):
    def __init__(self, ffn, num_chunks=8):
        super().__init__()
        self.ffn = ffn
        self.num_chunks = num_chunks

        # expose .net for compatibility
        if hasattr(ffn, "net"):
            self.net = ffn.net
    
    def set_ffn(self, ffn):
        self._ffn_ref = ffn

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape

        if seq_len < self.num_chunks * 100:
            return self.ffn(x)

        chunk_size = max(1, (seq_len + self.num_chunks - 1) // self.num_chunks)

        outputs = []
        for i in range(0, seq_len, chunk_size):
            chunk = x[:, i:i + chunk_size, :]
            outputs.append(self.ffn(chunk))

        return torch.cat(outputs, dim=1)


class TensorParallelV3Pipeline:
    """Safe FFN chunking pipeline."""
    
    _instances: Dict[int, 'TensorParallelV3Pipeline'] = {}
    
    def __init__(self, model: nn.Module, ffn_chunks: int = 8, verbose: int = 1):
        self.model = model
        self.model_id = id(model)
        self.ffn_chunks = ffn_chunks
        self.verbose = verbose
        
        self.wrappers: Dict[str, ChunkedFFN] = {}
        self.originals: Dict[str, nn.Module] = {}
        
        TensorParallelV3Pipeline._instances[self.model_id] = self
    
    @classmethod
    def get_instance(cls, model: nn.Module) -> Optional['TensorParallelV3Pipeline']:
        return cls._instances.get(id(model))
    
    def setup(self) -> Dict[str, Any]:
        info = {"ffn_wrapped": 0, "ffn_found": 0}
        
        if self.verbose >= 1:
            logger.info(f"TensorParallelV3: Setting up FFN chunking...")
            logger.info(f"  Chunks: {self.ffn_chunks}")
        
        # Find FFN modules
        ffn_modules = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'net') and isinstance(module.net, nn.Sequential):
                if name.endswith('.ff') or 'ff' in name.split('.')[-1]:
                    ffn_modules[name] = module
        
        info["ffn_found"] = len(ffn_modules)
        
        if self.verbose >= 1:
            logger.info(f"  Found {len(ffn_modules)} FFN modules")
        
        for name, original_ffn in ffn_modules.items():
            try:
                wrapper = ChunkedFFN(original_ffn, self.ffn_chunks)
                wrapper.set_ffn(original_ffn)
                
                self.originals[name] = original_ffn
                self.wrappers[name] = wrapper
                self._replace_module(name, wrapper)
                info["ffn_wrapped"] += 1
                
            except Exception as e:
                if self.verbose >= 1:
                    logger.warning(f"  Could not wrap {name}: {e}")
        
        if self.verbose >= 1:
            logger.info(f"  Wrapped {info['ffn_wrapped']} FFN modules")
        
        return info
    
    def _replace_module(self, path: str, new_module: nn.Module):
        parts = path.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def cleanup(self):
        for path, original in self.originals.items():
            self._replace_module(path, original)
        self.wrappers.clear()
        self.originals.clear()
        TensorParallelV3Pipeline._instances.pop(self.model_id, None)


class TensorParallelV3Node:
    """Safe FFN chunking - works with ALL modes including offloading."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ffn_chunks": ("INT", {"default": 8, "min": 2, "max": 32}),
                "verbose": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "setup"
    CATEGORY = "multigpu/experimental"
    
    def setup(self, model, ffn_chunks, verbose):
        lines = ["Tensor Parallel V3 - Safe FFN Chunking"]
        lines.append("=" * 50)
        
        try:
            target = model
            if hasattr(model, 'model'):
                target = model.model
                if hasattr(target, 'diffusion_model'):
                    target = target.diffusion_model
            
            existing = TensorParallelV3Pipeline.get_instance(target)
            if existing:
                lines.append("Already active!")
                return (model, "\n".join(lines))
            
            pipeline = TensorParallelV3Pipeline(
                model=target,
                ffn_chunks=ffn_chunks,
                verbose=verbose,
            )
            
            info = pipeline.setup()
            
            lines.append(f"FFN modules found: {info['ffn_found']}")
            lines.append(f"FFN modules wrapped: {info['ffn_wrapped']}")
            lines.append(f"Chunks: {ffn_chunks}")
            lines.append("")
            lines.append("This reduces FFN peak memory.")
            lines.append("Safe with offloading mode.")
            
        except Exception as e:
            lines.append(f"ERROR: {e}")
            import traceback
            lines.append(traceback.format_exc())
        
        return (model, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "TensorParallelV3Node": TensorParallelV3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorParallelV3Node": "Tensor Parallel V3 (Safe FFN Chunking)",
}
