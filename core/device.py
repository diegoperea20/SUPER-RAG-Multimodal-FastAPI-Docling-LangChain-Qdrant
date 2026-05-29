import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Detecta GPU CUDA o fallback a CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detectada: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = "cpu"
        logger.warning("Sin GPU. Usando CPU — BLIP sera mas lento.")
    return device
