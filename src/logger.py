# @ FileName: logger.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
import sys
import time
import functools
from os.path import join, exists
from datetime import datetime
import logging
import logging.handlers
from src import config, tool

pblog = None


def get_pblog(*args, **kwargs):
    global pblog
    if pblog is None:
        pblog = ProgressBarLog(*args, **kwargs)
    return pblog

def log_decorator(call_fn):
    @functools.wraps(call_fn)
    def log(self, *args, move=False):
        if self.current < self.total:
            sys.stdout.write(' ' * (self.width + 26) + '\r')
            sys.stdout.flush()
        call_fn(self, *args)
        if move:
            self._current += 1
            temp = datetime.now()
            delta = temp - self.last_time
            self.last_time = temp
            temp = temp + delta * (self.total - self.current)
            self.ok_time = str(temp).split('.')[0]
        if self.current < self.total:
            progress = int(self.width * self.current / self.total)
            temp = '{:2}%][{}]\r'.format(int(100 * self.current / self.total),
                                         self.ok_time)
            sys.stdout.write('[' + '=' * progress + '>' + '-' * (
                    self.width - progress - 1) + temp)
            sys.stdout.flush()

    return log


class ProgressBarLog:
    def __init__(self, total=50, width=76, current=0, logger=None):
        self.width = width - 26
        self.total = total
        self._current = current
        if logger is None:
            log_path = join(config.log_dir, config.cmd)
            desc = '{}_{}_{}_{}'. \
                format(config.dataset, config.model, config.action, config.desc)
            self.logger = gen_logger(gen_t_name(log_path, desc, '.log'))
        else:
            self.logger = logger
        self.last_time = datetime.now()
        self.ok_time = None
        self.pb_last_time = time.time()
        self.pb_begin_time = self.pb_last_time

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        if not isinstance(value, int):
            raise ValueError
        if value < 0 or value > self.total:
            raise ValueError
        self._current = value

    @log_decorator
    def debug(self, msg):
        self.logger.debug(msg)

    @log_decorator
    def info(self, msg):
        self.logger.info(msg)

    @log_decorator
    def warning(self, msg):
        self.logger.warning(msg)

    @log_decorator
    def error(self, msg):
        self.logger.error(msg)

    @log_decorator
    def exception(self, msg):
        self.logger.exception(msg)

    @log_decorator
    def print(self, *args):
        print(*args)

    @log_decorator
    def refresh(self):
        pass

    def pb(self, current, total, msg):
        TOTAL_BAR_LENGTH = 45.
        # _, term_width = os.popen('stty size', 'r').read().split()
        term_width = 94
        if current == 0:
            self.pb_begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH * current / total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for _ in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for _ in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        self.pb_last_time = cur_time
        tot_time = cur_time - self.pb_begin_time

        L = []
        L.append('Tot: %s' % format_time(tot_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current + 1, total))

        if current < total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()


def gen_logger(file_path, log_name=None):
    if log_name is None:
        log_name = config.log_name
    cmd_fmt = '[%(asctime)s] @%(name)s %(levelname)-8s%(message)s'
    cmd_datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = ColoredFormatter(cmd_fmt, cmd_datefmt)
    file_handler = logging.FileHandler(file_path)
    file_handler.formatter = formatter
    console_handler = logging.StreamHandler()
    console_handler.formatter = formatter
    logger = logging.getLogger(log_name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def gen_t_name(base_dir, desc, ext):
    tool.check_mkdir(base_dir)
    while True:
        dt = datetime.now()
        temp = join(base_dir, desc + dt.strftime('_%y%m%d_%H%M%S') + ext)
        if exists(temp):
            time.sleep(0.000001)
        else:
            break
    return temp


class ColoredFormatter(logging.Formatter):
    '''A colorful formatter.'''

    def __init__(self, fmt=None, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # Color escape string
        COLOR_RED = '\033[1;31m'
        COLOR_GREEN = '\033[1;32m'
        COLOR_YELLOW = '\033[1;33m'
        COLOR_BLUE = '\033[1;34m'
        COLOR_PURPLE = '\033[1;35m'
        COLOR_CYAN = '\033[1;36m'
        COLOR_GRAY = '\033[1;37m'
        COLOR_WHITE = '\033[1;38m'
        COLOR_RESET = '\033[1;0m'
        # Define log color
        LOG_COLORS = {
            'DEBUG': COLOR_BLUE + '%s' + COLOR_RESET,
            'INFO': COLOR_GREEN + '%s' + COLOR_RESET,
            'WARNING': COLOR_YELLOW + '%s' + COLOR_RESET,
            'ERROR': COLOR_RED + '%s' + COLOR_RESET,
            'CRITICAL': COLOR_RED + '%s' + COLOR_RESET,
            'EXCEPTION': COLOR_RED + '%s' + COLOR_RESET,
        }
        level_name = record.levelname
        msg = logging.Formatter.format(self, record)
        return LOG_COLORS.get(level_name, '%s') % msg


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f