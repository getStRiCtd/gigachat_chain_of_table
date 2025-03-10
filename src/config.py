import warnings

from enum import StrEnum
from pathlib import Path
from environs import Env
from dataclasses import dataclass

class GigaModels(StrEnum):
    GIGACHAT_LITE = 'Gigachat'
    GIGACHAT_PRO = 'GigaChat-Pro'
    GIGACHAT_MAX = 'GigaChat-Max'

class GigaScopes(StrEnum):
    PERS = 'GIGACHAT_API_PERS'
    B2B = "GIGACHAT_API_B2B"
    CORP = "GIGACHAT_API_CORP"

env = Env()

project_dir = Path(__file__).parent.parent
env_path = project_dir.joinpath('deployment', '.env')

if env_path.exists():
    env.read_env(path=env_path)
else:
    warnings.warn(f"Environment file not found at {env_path}. Using system Environment.")



@dataclass(frozen=True)
class GigaConfig:
    GIGACHAT_TOKEN = env.str("GIGACHAT_TOKEN")


giga_config = GigaConfig()