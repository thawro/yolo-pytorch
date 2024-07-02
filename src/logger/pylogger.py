import io
import logging
import re
import warnings

from colorlog.escape_codes import escape_codes

from src.utils.training import get_rank

fmt = "%(asctime)s %(levelname)s %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
path_regex = r"(\/.*?\.[\w:]+)"


class CustomFormatter(logging.Formatter):
    """Logging colored formatter based on https://stackoverflow.com/a/56944256/3638629.

    Handles FileHandler and StreamHandler logging objects.
    """

    _reset = "\x1b[0m"
    _msg = "%(message)s"
    _level = "%(levelname)s"

    def __init__(
        self,
        fmt: str = fmt,
        datefmt: str = datefmt,
        is_file: bool = False,
        datefmt_color: str | None = "light_black",
        url_color: str | None = "purple",
    ):
        if is_file:
            datefmt_color = None
            url_color = None
        if datefmt_color is not None:
            fmt = fmt.replace("%(asctime)s", f"{escape_codes[datefmt_color]} %(asctime)s")
        super().__init__(fmt, datefmt)

        for keyword in escape_codes.keys():
            fmt = fmt.replace(f"%({keyword})s", escape_codes[keyword])

        self.fmt = fmt
        self.datefmt = datefmt
        self.is_file = is_file
        self.url_color = url_color

        self.FORMATS = {
            logging.DEBUG: self.add_color_to_levelname(self.fmt, escape_codes["light_cyan"]),
            logging.INFO: self.add_color_to_levelname(self.fmt, escape_codes["green"]),
            logging.WARNING: self.add_color_to_levelname(self.fmt, escape_codes["yellow"]),
            logging.ERROR: self.add_color_to_levelname(self.fmt, escape_codes["red"]),
            logging.CRITICAL: self.add_color_to_levelname(self.fmt, escape_codes["bg_bold_red"]),
        }
        names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        max_len = max([len(name) for name in names.values()])
        num_spaces = {level_id: max_len - len(names[level_id]) for level_id in names}
        # spaces to the left
        # self.LEVEL_NAMES = {
        #     level_id: f"{' ' * num_spaces[level_id]}{names[level_id]}" for level_id in names
        # }
        # centered with spaces around
        self.LEVEL_NAMES = {
            level_id: f"{names[level_id].center(2 + len(names[level_id]) + num_spaces[level_id])}"
            for level_id in names
        }
        self._set_device("cpu", False)

    def _set_device(self, device: str, device_id: int):
        self.device = device
        self.device_id = device_id

    @property
    def device_info(self) -> str:
        return f"[{self.device.center(7)}] "

    @property
    def rank_info(self) -> str:
        return f"[rank={get_rank()}]"

    @classmethod
    def add_color_to_levelname(cls, fmt: str, color: str):
        return fmt.replace(
            f"{cls._level} {cls._msg}", f"{color}{cls._level}{cls._reset} {cls._msg}"
        )

    @classmethod
    def add_color_to_regex(cls, record: logging.LogRecord, regex: str, color: str):
        color_code = escape_codes[color]
        record.msg = re.sub(regex, rf"{color_code}\1{cls._reset}", record.msg)

    def format(self, record: logging.LogRecord):
        if self.is_file:
            log_fmt = self.fmt  # no formating for files
        else:
            log_fmt = self.FORMATS.get(record.levelno)
            if self.url_color is not None:
                if isinstance(record.msg, str):
                    self.add_color_to_regex(record, url_regex, self.url_color)

        record.levelname = self.LEVEL_NAMES[record.levelno]
        if isinstance(record.msg, str) and self.device_info not in record.msg:
            record.msg = self.rank_info + self.device_info + record.msg
        if self.is_file:
            for code in escape_codes.values():
                if not isinstance(record.msg, Exception) and code in record.msg:
                    record.msg = record.msg.replace(code, "")
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)


def get_cmd_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize command line logger"""
    formatter = CustomFormatter(
        fmt=fmt,
        datefmt=datefmt,
        is_file=False,
        datefmt_color="light_black",
        url_color="purple",
    )
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def get_file_pylogger(filepath: str, name: str = __name__) -> logging.Logger:
    """Initialize .log file logger"""
    formatter = CustomFormatter(fmt=fmt, datefmt=datefmt, is_file=True)
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(filepath, "a+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def remove_last_logged_line(file_log: logging.Logger):
    """Remove the last line of log file"""
    file: io.TextIOWrapper = file_log.handlers[0].stream
    file.seek(0)
    lines = file.readlines()
    file.seek(0)
    file.truncate()
    file.writelines(lines[:-1])
    file.seek(0, 2)


def log_breaking_point(
    msg: str,
    n_top: int = 0,
    n_bottom: int = 0,
    top_char: str = "-",
    bottom_char: str = "-",
    num_chars: int = 100,
    worker: int | str = 0,
):
    rank = get_rank()
    if worker == "all" or worker == rank:
        TOP_LINE = top_char * num_chars
        BOTTOM_LINE = bottom_char * num_chars
        for _ in range(n_top):
            log.info(TOP_LINE)
        log.info(msg.center(num_chars))
        for _ in range(n_bottom):
            log.info(BOTTOM_LINE)


def showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file=None,
    line: str | None = None,
):
    message = warnings.formatwarning(message, category, filename, lineno, line)
    log.warning(message)


def log_msg(msg: str, level: int = logging.INFO, rank: int = 0):
    if get_rank() == rank:
        level2fn[level](msg)


log = get_cmd_pylogger(__name__)

level2fn = {
    logging.INFO: log.info,
    logging.WARN: log.warning,
    logging.ERROR: log.error,
    logging.CRITICAL: log.critical,
    logging.DEBUG: log.debug,
}

warnings.showwarning = showwarning
