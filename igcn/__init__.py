from .igabor import GaborFunction, gabor, gabor_gradient
from .modules import IGConv, IGabor, IGBranched, IGParallel, MaxGabor
from .models import IGCN

import logging
from logging.config import dictConfig

logging_config = {
    "version": 1,
    "formatters": {
        "f": {
            "format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s (%(filename)s:%(lineno)s)"
        }
    },
    "handlers": {
        "h": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "f",
            "level": logging.DEBUG,
            "filename": "igcn.log",
        },
    },
    "root": {"handlers": ["h"], "level": logging.DEBUG,},
}
dictConfig(logging_config)
