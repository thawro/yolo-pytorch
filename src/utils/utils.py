"""Other utilities"""

import random
from contextlib import contextmanager
from datetime import datetime
from timeit import default_timer

import numpy as np
from colorlog.escape_codes import escape_codes

_norm = list[float] | np.ndarray


def random_float(min: float, max: float) -> float:
    return random.random() * (max - min) + min


def get_current_date_and_time(format: str = "%m-%d_%H:%M") -> str:
    now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H:%M")
    dt_string = now.strftime(format)
    return dt_string


def prepend_exception_message(exception: Exception, prefix: str):
    _args = exception.args
    if len(_args) >= 1:
        exception.args = (f"{prefix}{_args[0]}", *_args[1:])


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def list2array(x: _norm) -> np.ndarray:
    if isinstance(x, list):
        return np.array(x)
    return x


def colorstr(txt: str, color: str = "bold_blue") -> str:
    reset = "\x1b[0m"
    return escape_codes[color] + txt + reset
