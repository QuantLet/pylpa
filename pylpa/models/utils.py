from pylpa.models.arch import ARCH
from pylpa.models.armagarch import ARMAGARCH
from pylpa.models.garch import GARCH


def build_model_from_config(config):
    if config["name"] == "arch":
        model = ARCH(**config["params"])
    elif config["name"] == "garch":
        model = GARCH(**config["params"])
    elif config["name"] == "arma-garch":
        model = ARMAGARCH(**config["params"])
    else:
        raise NotImplementedError(config["name"])
    return model