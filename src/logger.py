import logging
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# Try to import ipynbname for general Jupyter support
try:
    import ipynbname
except ImportError:
    ipynbname = None

class JSONFormatter(logging.Formatter):
    def format(self, record):
        file_origin = record.filename
        
        # 1. Check if running in VS Code Jupyter
        # VS Code injects '__vsc_ipynb_file__' into the global scope
        vsc_file = sys.modules['__main__'].__dict__.get('__vsc_ipynb_file__')
        
        if vsc_file:
            file_origin = Path(vsc_file).name
        # 2. Check if running in standard Jupyter via ipynbname
        elif ipynbname:
            try:
                file_origin = ipynbname.name() + ".ipynb"
            except:
                pass
        
        # 3. Clean up the "12345.py" temp names if still present
        if file_origin.endswith('.py') and file_origin[:-3].isdigit():
            file_origin = "Jupyter_Notebook_Unknown"

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "file": file_origin,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

# --- DYNAMIC ROOT DIRECTORY ---
# Gets the root by going up from src/logger.py
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

# Setup
logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.INFO)

# Only add handlers if they don't exist (prevents duplicate logs in notebooks)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)