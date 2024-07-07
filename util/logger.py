import logging


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, filename, level='info',
                 fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s : %(message)s',
                 datefmt='%Y-%m-%d %H:%M'):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt=fmt, datefmt=datefmt)

        fh = logging.FileHandler(filename=filename, mode='a')
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str)
        self.logger.addHandler(fh)