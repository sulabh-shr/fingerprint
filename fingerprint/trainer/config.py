import yaml


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
            # value = {value}\''  # string representation of list
            value = convert_str_to_list(value)
            element_type = type(prev_value[0])
            value = [element_type(i) for i in value]

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


def convert_str_to_list(list_as_string: str):
    list_as_string = list_as_string.strip()
    list_as_string = list_as_string.strip('[]()')

    result = []

    for i in list_as_string.split(','):
        i = i.strip()
        result.append(i)
    return result
