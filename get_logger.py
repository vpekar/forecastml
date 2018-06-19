import logging
import logging.handlers

logging.basicConfig(level=logging.DEBUG)


def get_logger(name, fname):

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False

    format = "%(asctime)-15s:%(name)s:%(levelname)s:%(message)s"
    formatter = logging.Formatter(fmt=format)

    # file with actual log
    file_handler = logging.FileHandler(fname, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # file with main results
    file_handler = logging.FileHandler(fname.replace('.', '-main.'), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
