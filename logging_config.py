# logging_config.py

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def configure_logging():
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s: %(message)s')

    log_filename = f"log_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)
