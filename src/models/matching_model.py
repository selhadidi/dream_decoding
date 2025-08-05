import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, input_shape=(60, 2001), n_classes=13, dropout=0.25):
        super(EEGNet, self).__init__()
        self.n_classes = n_classes
        self.dropout = dropout
        self.in_channels, self.time_samples = input_shape
        assert len(input_shape) == 2, f"Expected input_shape to be (channels, timepoints), got {input_shape}"

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # Block 2: Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(8, 16, kernel_size=(self.in_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(dropout)

        # Block 3: Separable Convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )

        # Final classification
        self.classifier = nn.Linear(self._calculate_flatten_dim(), n_classes)

    def _calculate_flatten_dim(self):
        dummy_input = torch.zeros(1, 1, self.in_channels, self.time_samples)
        x = self.conv1(dummy_input)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.separable_conv(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ------------------- Simple CNN -------------------
class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(8, 8), channels=1, n_classes=13, dropout=0.3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        flat_size = (input_shape[0] // 2) * (input_shape[1] // 2) * 32
        self.fc1 = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        #x = x.unsqueeze(1)  # [B, 1, H, W]
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ------------------- Simple MLP -------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512], output_dim=384, dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ------------------- Classic Model Wrapper -------------------
class ClassicModelWrapper:
    def __init__(self, model_type="xgboost", task_type="classification", **kwargs):
        if model_type == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            self.model = XGBClassifier(**kwargs) if task_type == "classification" else XGBRegressor(**kwargs)

        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor
            if task_type == "classification":
                self.model = LGBMClassifier(**kwargs)
            else:
                base = LGBMRegressor(**kwargs)
                self.model = MultiOutputRegressor(base)
                
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            self.model = RandomForestClassifier(**kwargs) if task_type == "classification" else RandomForestRegressor(**kwargs)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)

    def get_params(self):
        return self.model.get_params()

# ------------------- Get Model -------------------
def get_model(model_type="mlp", model_config=None, input_dim=None, output_dim=None, task_type=None):
    """
    Initialize and return a model based on type and configuration.
    
    Args:
        model_type (str): One of 'mlp', 'cnn', 'eegnet', or classic model keys like 'xgb_default'.
        model_config (dict): Model-specific hyperparameters (from MODEL_CONFIGS or CLASSIC_MODEL_CONFIGS).
        input_dim (int): Input dimension (for MLP).
        output_dim (int): Output dimension or number of classes (for classification) or vector size (for regression).
        task_type (str): "classification" or "regression" — required for classic models.

    Returns:
        An initialized model instance.
    """
    if model_type == "mlp":
        hidden_dims = model_config.get("hidden_dims", [512, 512])
        dropout_rate = model_config.get("dropout_rate", 0.3)
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )

    elif model_type == "cnn":
        shape = model_config.get("input_shape", (60, 2001))
        return SimpleCNN(
            input_shape=shape,
            channels=model_config.get("channels", 1),
            n_classes=output_dim,
            dropout=model_config.get("dropout_rate", 0.3)
        )

    elif model_type == "eegnet":
        shape = model_config.get("input_shape", (60, 2001))
        return EEGNet(
            input_shape=shape,
            n_classes=output_dim,
            dropout=model_config.get("dropout_rate", 0.4)
        )

    elif model_type in ["xgboost", "lightgbm", "random_forest"]:
        # Classic ML model wrapper
        return ClassicModelWrapper(
            model_type=model_type,
            task_type=task_type or "classification",
            **model_config.get("params", {})
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
