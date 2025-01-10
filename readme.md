# TorchOptimizer Class

This repository contains the `TorchOptimizer` class, a tool designed for efficient hyperparameter optimization of PyTorch Lightning models using Bayesian Optimization via scikit-optimize. It performs multithreaded optimization by exploring a defined hyperparameter search space and evaluating configurations against a specified performance metric.

---

## Features

-   **Bayesian Optimization**: Utilizes Gaussian Processes to model the objective function and optimize hyperparameters.
-   **Parallel Evaluation**: Executes multiple evaluations simultaneously for faster results.
-   **Integration with PyTorch Lightning**: Directly optimizes PyTorch Lightning models, making it compatible with the Lightning ecosystem.
-   **Flexible Search Space**: Supports defining complex hyperparameter search spaces with constraints.

---

## How It Works

### Gaussian Processes in Hyperparameter Optimization

Gaussian Processes (GPs) are used to approximate the objective function (e.g., validation loss) over the hyperparameter space. They work by:

1. **Modeling Uncertainty**: GPs provide both a mean prediction and uncertainty, enabling exploration of less certain areas.
2. **Bayesian Updating**: The GP model updates its beliefs about the objective function as new hyperparameter evaluations are added.
3. **Acquisition Function**: Determines the next hyperparameter set to evaluate by balancing exploration (searching unknown areas) and exploitation (refining known good regions).

This process allows for efficient exploration of high-dimensional and expensive-to-evaluate search spaces.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Define Your Model and Search Space

Before using `TorchOptimizer`, define your PyTorch Lightning model and specify the hyperparameter search space using `skopt.space.Dimension`.

### Example

```python
import torch
import pytorch_lightning as pl
from skopt.space import Real, Integer, Categorical
from torch.utils.data import DataLoader
from torch_optimizer import TorchOptimizer

# Define your PyTorch Lightning model
class MyModel(pl.LightningModule):
    def __init__(self, lr, hidden_size):
        super().__init__()
        self.lr = lr
        self.hidden_size = hidden_size
        self.layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.layer(x)

    # Define training step
    def training_step(batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    # Define validation step
    def validation_step(batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Set up dataloaders
train_loader = DataLoader(torch.randn(100, 10), batch_size=32)
val_loader = DataLoader(torch.randn(20, 10), batch_size=32)

# Define hyperparameter search space
space = [
    Real(1e-4, 1e-2, name="lr"),
    Integer(32, 256, name="hidden_size")
]

# Define the trainer parameters
trainer_args = {
    "max_epochs": 100,
    "accelerator": "gpu",
    "devices": [0],
    "log_every_n_steps": 1,
    "num_sanity_val_steps": 0,
}

# Define constraints
def constraint(params):
    return params["hidden_size"] % 32 == 0



# Instantiate the optimizer
optimizer = TorchOptimizer(
    model=MyModel,
    trainer_args=trainer_args,
    train_loader=train_loader,
    val_loader=val_loader,
    monitor="val_loss",
    maximise=False,
    space=space,
    constraint=constraint,
    n_calls=50,
    n_initial_points=10
)

# Run the optimizer
best_params = optimizer()
print("Best Parameters:", best_params)
```

### Parameters

-   `model`: Your PyTorch Lightning model.
-   `trainer_args`: Arguments for the PyTorch Lightning Trainer.
-   `train_loader`, `val_loader`: DataLoaders for training and validation.
-   `monitor`: Metric to evaluate during optimization (e.g., `val_loss`).
-   `maximise`: Boolean to indicate whether to maximize or minimize the metric.
-   `space`: List of hyperparameter dimensions (`skopt.space.Dimension`).
-   `constraint`: Callable to define valid hyperparameter configurations.
-   `n_calls`: Total number of optimization iterations.
-   `n_initial_points`: Number of random points sampled before optimization begins.

---

## License

This project is licensed under the MIT License.
