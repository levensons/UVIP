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
            "budget": 2500, #2500
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
            "lr": 3e-4,
            "weight_decay": 1e-4
        }
    )

    config.fqe_params = config_dict.ConfigDict(
        {
            "n_epochs": 250,
            "hidden_size": 256,
            "bs": 256
        }
    )

    config.gamma = 0.99
    config.device = device
    config.seed = 42 # 3, 9, 
    config.T = 1000
    config.H = 100

    config.alg_type = alg_type

    return config