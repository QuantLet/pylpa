import logging

import coloredlogs

LOGGER_LEVEL = "info"


def get_logger(name, level="debug", save_path=None):
    logger = logging.getLogger(name)
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    elif level == "error":
        level = logging.ERROR
    elif level == "notset":
        level = logging.NOTSET
    else:
        raise NotImplementedError()
    logger.setLevel(level)

    stdout_logger = logging.StreamHandler()
    stdout_logger.setFormatter(
        logging.Formatter(
            "[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s"
        )
    )

    logger.addHandler(stdout_logger)
    logger.propagate = False
    coloredlogs.install(
        logger=logger,
        level=level,
        fmt="[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s",
    )

    if save_path:
        fh = logging.FileHandler(save_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        LOGGER.addHandler(fh)

    return logger


LOGGER = get_logger("LPA", level=LOGGER_LEVEL)
