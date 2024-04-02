import ast
import yaml
import easydict


def load_cfg(cfgs, opts=None):
    cfg = {}
    for path in cfgs:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))
    if opts is not None:
        cfg = merge_from_list(cfg, opts)
    return cfg


def merge_from_list(cfg, opts):
    assert len(opts) % 2 == 0

    for idx in range(0, len(opts) - 1, 2):
        parameter = opts[idx]
        prev_value = access_nested(cfg, parameter)
        value = opts[idx + 1]

        if isinstance(prev_value, (int, float)):
            value = type(prev_value)(value)
        elif isinstance(prev_value, bool):
            if value in ('true', 'True'):
                value = True
            elif value in ('false', 'False'):
                value = False
            else:
                raise ValueError(f'Invalid value {value} for boolean parameter {parameter}')
        elif value in ('None', 'none'):
            value = None
        elif isinstance(prev_value, (tuple, list)):
            value = f'\'{value}\''  # string representation of list
            value = ast.literal_eval(value)

        set_nested(cfg, parameter, value)

    return cfg


def access_nested(d, key):
    keys = key.split('.')
    for k in keys:
        d = d[k]
    return d


def set_nested(d, key, value):
    keys = key.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return d
