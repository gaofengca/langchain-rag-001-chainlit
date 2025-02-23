import logging
import logging.config
import json
import os


class Logger:
    def __init__(self):
        self.resource_path = os.path.join(os.path.dirname(__file__), '../../resources/logging_config.json')
        self.setup_logging()

    def setup_logging(self):
        # Load logging configuration from a JSON file
        with open(self.resource_path, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)

    @staticmethod
    def get_logger():
        return logging.getLogger()
