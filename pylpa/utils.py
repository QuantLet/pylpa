def default_config(config: dict):
    if 'save_k' not in config.keys():
        config['save_k'] = False
    if 'solver' not in config.keys():
        config['solver'] = 'SLSQP'
    if 'maxiter' not in config.keys():
        config['maxiter'] = 100
    if 'generate' not in config.keys():
        config['generate'] = 'normal'

    config['model'] = default_model_config(config['model'])

    return config


def default_model_config(config: dict):
    if config['name'] == 'arma-arch':
        if 'params' not in config.keys():
            config["params"] = {'arma': True, 'garch': False}
        if 'type' not in config.keys():
            config["type"] = "arch"
    elif config['name'] == 'arma-garch':
        if 'params' not in config.keys():
            config["params"] = {'arma': True, 'garch': True}
        if 'type' not in config.keys():
            config["type"] = "garch"
    elif config['name'] == 'garch':
        if 'params' not in config.keys():
            config["params"] = {'arma': False, 'garch': True}
        if 'type' not in config.keys():
            config["type"] = "garch"
    elif config['name'] == 'arch':
        if 'params' not in config.keys():
            config["params"] = {'arma': False, 'garch': False}
        if 'type' not in config.keys():
            config["type"] = "arch"
    elif config['name'] in ['carlInd', 'carlVol']:
        config["type"] = config['name']

    return config
