import ConfigParser

__author__ = 'amanda'


class Params:
    def __init__(self, config_path):
        cfg = ConfigParser.ConfigParser()
        cfg.read(config_path)
        for cur_sec in cfg.sections():
            for cur_param in cfg.items(cur_sec):
                setattr(self, cur_param[0], eval(cur_param[1]))