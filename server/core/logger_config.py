import os
from pathlib import Path
from typing import Final

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
LOG_RETENTION_DAYS: Final[str] = os.getenv("LOG_RETENTION_DAYS", "30 days")
LOG_ROTATION_SIZE: Final[str] = os.getenv("LOG_ROTATION_SIZE", "100 MB")

PROJECT_ROOT: Final[Path] = Path(__file__).parent
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(exist_ok=True)
for subdir in ['archive']:
    (LOG_DIR / subdir).mkdir(exist_ok=True)

LOG_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)