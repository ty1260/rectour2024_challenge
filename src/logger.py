import os
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG

def set_logger(log_dir, log_name, log_level=DEBUG):
    logger = getLogger(log_name)
    logger.setLevel(log_level)

    formatter = Formatter("%(asctime)s [%(levelname)s] (%(filename)s | %(funcName)s | %(lineno)s) %(message)s")

    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    file_handler = FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
