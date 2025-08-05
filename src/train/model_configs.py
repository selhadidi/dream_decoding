MODEL_CONFIGS = {
    "simple_512": {
        "hidden_dims": [512],
        "dropout_rate": 0.3,
    },
    "deep_512_512": {
        "hidden_dims": [512, 512],
        "dropout_rate": 0.3,
    },
    "deep_1024_512": {
        "hidden_dims": [1024, 512],
        "dropout_rate": 0.3,
    },
    "medium_256_256": {
        "hidden_dims": [256, 256],
        "dropout_rate": 0.2,
    },
    "big_1024_512_256": {
        "hidden_dims": [1024, 512, 256],
        "dropout_rate": 0.3,
    },
    "big_1024_512_512_256": {
        "hidden_dims": [1024, 512, 512, 256],
        "dropout_rate": 0.5,
    },
    "big_1024_1024_512_512": {
        "hidden_dims": [1024, 1024, 512, 512],
        "dropout_rate": 0.3,
    },
    "big_2048_1024_512": {
        "hidden_dims": [2048, 1024, 512],
        "dropout_rate": 0.3,
    },
    "big_512_512_512": {
        "hidden_dims": [512, 512, 512],
        "dropout_rate": 0.3,
    },
    "small_128": {
        "hidden_dims": [128],
        "dropout_rate": 0.1,
    },
    "small_128_128": {
        "hidden_dims": [128, 128],
        "dropout_rate": 0.2,
    },
    "dropout_0.5": {
        "hidden_dims": [1024, 512, 256],
        "dropout_rate": 0.5,
    },
    "wide_2048_2048": {
        "hidden_dims": [2048, 2048],
        "dropout_rate": 0.3,
    },
    "tiny_64": {
    "hidden_dims": [64],
    "dropout_rate": 0.1,
    },
    "tiny_64_64": {
        "hidden_dims": [64, 64],
        "dropout_rate": 0.1,
    },
    "small_128_64": {
        "hidden_dims": [128, 64],
        "dropout_rate": 0.2,
    },
    "small_128_32": {
        "hidden_dims": [128, 32],
        "dropout_rate": 0.2,
    },
    "mini_256": {
        "hidden_dims": [256],
        "dropout_rate": 0.2,
    },
    "mini_256_128": {
        "hidden_dims": [256, 128],
        "dropout_rate": 0.2,
    },
    "rising_128_512": {
        "hidden_dims": [128, 512],
        "dropout_rate": 0.2,
    },
    "rising_128_512_512": {
        "hidden_dims": [128, 512, 512],
        "dropout_rate": 0.3,
    },
    "rising_128_128_512": {
        "hidden_dims": [128, 128, 512],
        "dropout_rate": 0.3,
    },
    "jumping_128_1024": {
        "hidden_dims": [128, 1024],
        "dropout_rate": 0.3,
    },
    "long_128_128_128_128": {
        "hidden_dims": [128, 128, 128, 128],
        "dropout_rate": 0.3,
    },
    "cnn_small": {
        "dropout_rate": 0.25
    },
    "cnn_medium": {
        "input_shape": (1, 759),
        "channels": [32, 64],
        "kernel_sizes": [5, 5],
        "dropout_rate": 0.3
    },
    "cnn_large": {
        "input_shape": (1, 759),
        "channels": [64, 128],
        "kernel_sizes": [7, 5],
        "dropout_rate": 0.4
    },
    "eegnet_small": {
        "dropout_rate": 0.25,
    },
    
    "eegnet_medium": {
        "dropout_rate": 0.4,
    },
    
    "eegnet_large": {
        "dropout_rate": 0.5,
    }
}

CLASSIC_MODEL_CONFIGS = {
    "xgb_default": {
        "model_type": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "verbosity": 0
        }
    },
    "xgb_shallow": {
        "model_type": "xgboost",
        "params":
            {"n_estimators": 50,
            "max_depth": 7,
            "learning_rate": 0.05,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "verbosity": 0}
    },
    "lightgbm_default": {
        "model_type": "lightgbm",
        "params":{
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "verbosity": -1}
    },
    "rf_default": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42}
    },
    "rf_shallow": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 50,
            "max_depth": 10,
            "random_state": 42}
    },
    "rf_v_shallow": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 75,
            "max_depth": 7,
            "random_state": 42}
    },
    "rf_m_shallow": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 75,
            "max_depth": 7,
            "random_state": 42}
    },
    "rf_s_shallow": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 75,
            "max_depth": 5,
            "random_state": 42}
    },
    "rf_l_shallow": {
        "model_type": "random_forest",
        "params":{
            "n_estimators": 50,
            "max_depth": 8,
            "random_state": 42}
    },
    "lightgbm_tuned": {
        "model_type": "lightgbm",
        "params": {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
            "verbosity": -1
        }
    },
    "rf_deep": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 50,
            "random_state": 42
        }
    },
    "xgb_regularized": {
        "model_type": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "verbosity": 0
        }
    }
}