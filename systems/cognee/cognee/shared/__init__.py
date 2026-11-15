# Loading LLM Config from data_models.py requires to have dotenv imported first
# and to have it loaded

import dotenv
from pathlib import Path

_env_path = Path(__file__).resolve().parents[2] / ".env"
dotenv.load_dotenv(dotenv_path=_env_path, override=True)
