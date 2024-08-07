data2label = {"mrpc": 2, "poem": 4, "rte": 2, 'wnli':2, 
            "Fashion_mnist":10, "Mnist":10, "Cifar":10, "Cifar100": 100}

defense2mod = {"fedavg": "train_fedavg", 
                "rflpa": "train_rflpa",
                "brea": "train_brea", 
                "bulyan": "train_bulyan", 
                "trimmean": "train_trimmean",
                "cdp": "train_dp",
                "ldp": "train_dp",
                "rsa": "train_rsa",
                "gw": "train_gw",
                "pg": "train_pg",
                }

bd_attacks = ["badnet", "scaling"]