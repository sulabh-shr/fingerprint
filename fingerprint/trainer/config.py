import yaml


def load_cfg(cfgs):
    cfg = {}
    for path in cfgs:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    return cfg
