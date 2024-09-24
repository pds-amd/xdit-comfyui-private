import torch
from logging import getLogger

logger = getLogger(__name__)

def check_flash_attn():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            return False
        else:
            from flash_attn import flash_attn_func
            from flash_attn import __version__

            if __version__ < "2.6.0":
                raise ImportError(f"install flash_attn >= 2.6.0")
            return True
    except ImportError:
        logger.warning(
            f'Flash Attention library "flash_attn" not found, '
            f"using pytorch attention implementation"
        )
        return False

HAS_FLASH_ATTN = check_flash_attn()