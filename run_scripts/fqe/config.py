from ml_collections import config_dict

def get_config(alg_type):
    device = "cpu"

    config_e = {
        "KBUCBVI": config_dict.ConfigDict(
            {
                "params": config_dict.ConfigDict(
                    {
                        "kernel_type": "gaussian",
                        "max_repr": 500,
                        "beta": 0.01,
                        "bandwidth": 0.025,
                        "min_dist": 0.05
                    }
                ),
                "budget": 5000, #5000
                "gamma": 0.99
            }
        )
    }[alg_type]

    config_b = config_dict.ConfigDict(
        {
            "params": config_dict.ConfigDict(
                {
                    "kernel_type": "gaussian",
                    "max_repr": 500,
                    "beta": 0.01,
                    "bandwidth": 0.025,
                    "min_dist": 0.05
                }
            ),
            "budget": 5000, #2500
            "gamma": 0.99
        }
    )

    config_env = config_dict.ConfigDict(
        {}
    )

    config = config_dict.ConfigDict(
        {
            "config_e": config_e,
            "config_b": config_b,
            "config_env": config_env
        }
    )

    config.optim_conf = config_dict.ConfigDict(
        {
            "lr": 0.0003,
            "weight_decay": 1e-4
        }
    )

    config.fqe_params = config_dict.ConfigDict(
        {
            "n_epochs": 400, #250
            "hidden_size": 32,
            "bs": 32
        }
    )

    config.gamma = 0.99
    config.device = device
    config.seeds = [3, 9, 33, 42, 1812] # [3, 9, 33, 42, 1812]
    config.T = 25
    config.H = 100

    config.alg_type = alg_type

    return config