# Import necessary libraries
import torch
from pytorch_lightning import LightningModule  # Base class for PyTorch Lightning models
from optimizer import TorchOptimizer          # Our custom optimizer class
from skopt.space import Real                  # For defining continuous hyperparameter ranges

# Define a simple PyTorch Lightning model for demonstration
class ExampleModel(LightningModule):
    def __init__(self, lr):
        # Initialize model with learning rate parameter
        super().__init__()
        self.lr = lr
        # Create a simple linear layer with 10 input features and 1 output
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        # Define the forward pass - simply apply the linear layer
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # Define what happens in one training step:
        x, y = batch                                          # Unpack the batch into inputs and targets
        y_hat = self(x)                                      # Get model predictions
        loss = torch.nn.functional.mse_loss(y_hat, y)        # Calculate MSE loss
        return loss                                          # Return the loss for optimization

    def validation_step(self, batch, batch_idx):
        # Define what happens in one validation step:
        x, y = batch                                          # Unpack the validation batch
        y_hat = self(x)                                      # Get model predictions
        val_loss = torch.nn.functional.mse_loss(y_hat, y)    # Calculate validation loss
        self.log("val_loss", val_loss)                       # Log the validation loss for tracking
        return val_loss

    def configure_optimizers(self):
        # Define which optimizer to use with current learning rate
        return torch.optim.SGD(self.parameters(), lr=self.lr)

# Define the hyperparameter search space
# Here we're only optimizing the learning rate between 0.0001 and 0.01
space = [Real(1e-4, 1e-2, name="lr")]

# Set up PyTorch Lightning Trainer configuration
trainer_args = {
    "max_epochs": 100,              # Train for 100 epochs
    "accelerator": "gpu",           # Use GPU acceleration
    "devices": [0],                 # Use first GPU
    "log_every_n_steps": 1,         # Log metrics every step
    "num_sanity_val_steps": 0,      # Skip validation sanity checks
}

# Create the TorchOptimizer instance with all necessary parameters
optimizer = TorchOptimizer(
    model=ExampleModel,                         # The model class to optimize
    trainer_args=trainer_args,                  # Training configuration
    train_loader=torch.randn(100, 10),         # Random training data (100 samples, 10 features)
    val_loader=torch.randn(20, 10),            # Random validation data (20 samples, 10 features)
    monitor="val_loss",                        # Metric to optimize
    maximise=False,                            # We want to minimize validation loss
    space=space,                               # Hyperparameter search space
    constraint=lambda x: True,                 # No constraints on hyperparameters
    n_calls=20,                                # Number of different configurations to try
    n_initial_points=5                         # Number of random configurations before optimization
)

# Run the optimization process
best_params = optimizer()                       # This runs the full optimization process
print("Optimized Parameters:", best_params)     # Print the best hyperparameters found