import torch
from pytorch_lightning import LightningModule
from optimizer import TorchOptimizer
from skopt.space import Real

# Define your model
class ExampleModel(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    # Define training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    # Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

# Define hyperparameter space
space = [Real(1e-4, 1e-2, name="lr")]

# Define the trainer parameters
trainer_args = {
    "max_epochs": 100,
    "accelerator": "gpu",
    "devices": [0],
    "log_every_n_steps": 1,
    "num_sanity_val_steps": 0,
}

# Instantiate optimizer
optimizer = TorchOptimizer(
    model=ExampleModel,
    trainer_args=trainer_args,
    train_loader=torch.randn(100, 10),
    val_loader=torch.randn(20, 10),
    monitor="val_loss",
    maximise=False,
    space=space,
    constraint=lambda x: True,
    n_calls=20,
    n_initial_points=5
)

# Run optimization
best_params = optimizer()
print("Optimized Parameters:", best_params)