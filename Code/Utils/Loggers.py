import logging
import sys
import os
from datetime import datetime


class PrintLogger:
    def write(self, message):
        if message.strip() != "":
            logging.info(message.strip())

    def flush(self):
        pass


class Logger:

    def __init__(self):
        os.makedirs("logs", exist_ok=True)

        log_name = datetime.now().strftime("logs/train_%Y%m%d_%H%M%S.log")

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(log_name)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        sys.stdout = PrintLogger()

        logging.info("Logger initialized")
        logging.info("Log file: %s", log_name)