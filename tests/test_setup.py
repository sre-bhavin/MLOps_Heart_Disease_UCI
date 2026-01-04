import os
import json
import pytest
import sys
from src.logger import logger, LOG_FILE_PATH
from src.exception import CustomException

def test_logger_creates_file():
    """Check if the log file is physically created in the logs/ folder."""
    logger.info("Testing pytest logger integration")
    assert os.path.exists(LOG_FILE_PATH)

def test_logger_json_format():
    """Verify that the last line of the log file is valid JSON and has required keys."""
    logger.info("JSON Format Check")
    
    with open(LOG_FILE_PATH, "r") as f:
        lines = f.readlines()
        last_log = json.loads(lines[-1])  # Parse the last line
    
    assert "timestamp" in last_log
    assert "level" in last_log
    assert "message" in last_log
    assert "file" in last_log

def test_custom_exception_details():
    """Ensure CustomException captures the correct filename and line number."""
    try:
        # Intentionally trigger an error
        a = 1 / 0
    except Exception as e:
        exc = CustomException(e, sys)
        error_msg = str(exc)
        
        # Check if our custom formatting is present
        assert "Error in" in error_msg
        assert "test_setup.py" in error_msg
        assert "line" in error_msg