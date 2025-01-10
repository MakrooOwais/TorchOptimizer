"""
This module implements the TorchOptimizer class, which performs multithreaded hyperparameter optimization for a given PyTorch Lightning model.

Hyperparameter tuning is crucial for optimizing model performance by searching the configuration space to find parameter values that maximize (or minimize) a chosen evaluation metric. This module uses Bayesian Optimization through scikit-optimize to efficiently explore this search space.
"""

import copy
import json
import pytorch_lightning as pl
import os
import contextlib
import torch

from functools import partial
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skopt import Optimizer
from skopt.space import Dimension
from multiprocessing import cpu_count
from joblib import Parallel, delayed


class TorchOptimizer:
    def __init__(
        self,
        model: pl.LightningModule,
        trainer_args: dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        monitor: str,
        maximise: bool,
        space: list[Dimension],
        constraint: callable,
        n_calls: int,
        n_initial_points: int,
    ):
        """
        Initializes the TorchOptimizer with the provided parameters.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to optimize.
            trainer_args (dict): Configuration arguments for the PyTorch Lightning Trainer.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            monitor (str): Metric to monitor during optimization (e.g., validation loss).
            maximise (bool): If True, the metric will be maximized; otherwise, minimized.
            space (list[Dimension]): List of hyperparameter search dimensions.
            constraint (callable): Function defining constraints for hyperparameters.
            n_calls (int): Number of optimization iterations.
            n_initial_points (int): Number of random points sampled before optimization.
        """
        self.model = model
        self.tr_loader, self.vl_loader = train_loader, val_loader
        self.trainer_args = trainer_args
        self.space = space
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.monitor = monitor
        self.max = maximise
        self.constraint = constraint
        self.objective = partial(
            TorchOptimizer._objective,
            model_class=self.model,
            trainer_args=self.trainer_args,
            tr_loader=self.tr_loader,
            vl_loader=self.vl_loader,
            monitor=self.monitor,
            max_=self.max,
        )
        self.names = [x.name for x in self.space]

    @staticmethod
    def _objective(
        params: list,
        model_class: pl.LightningModule,
        trainer_args: dict,
        tr_loader: torch.utils.data.DataLoader,
        vl_loader: torch.utils.data.DataLoader,
        names,
        monitor,
        max_,
    ) -> float:
        """
        Objective function to evaluate a set of hyperparameters.

        Args:
            params (list): List of hyperparameter values.
            model_class (pl.LightningModule): Class of the PyTorch Lightning model.
            trainer_args (dict): Arguments for the PyTorch Lightning Trainer.
            tr_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            vl_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            names (list): Names of the hyperparameters being optimized.
            monitor (str): Metric to evaluate.
            max_ (bool): Whether to maximize or minimize the metric.

        Returns:
            float: The evaluation result for the current hyperparameter set.
        """

        save_dir = "_".join([str(x) for x in list(params)])
        model_args = dict(zip(names, params))
        os.makedirs("logs" + save_dir, exist_ok=True)

        with open(f"logs/{save_dir}/out.txt", "w") as f, contextlib.redirect_stdout(
            f
        ), contextlib.redirect_stderr(f):
            if "logger" not in trainer_args.keys():
                trainer_args["logger"] = TensorBoardLogger(save_dir=f"logs/{save_dir}")
            if "callbacks" not in trainer_args.keys():
                trainer_args["callbacks"] = [
                    ModelCheckpoint(
                        dirpath=trainer_args["logger"].save_dir,
                        monitor="val_diff",
                        mode="max",
                        save_top_k=1,
                        filename="best_model",
                    )
                ]
            if "enable_progress_bar" not in trainer_args:
                trainer_args["enable_progress_bar"] = True

            model = model_class(*params)
            trainer = pl.Trainer(**trainer_args)

        return TorchOptimizer.train(
            model,
            model_class,
            model_args,
            trainer,
            trainer_args,
            copy.deepcopy(tr_loader),
            copy.deepcopy(vl_loader),
            monitor,
            max_,
        )

    @staticmethod
    def train(
        model: pl.LightningModule,
        model_class: pl.LightningModule,
        model_args: dict,
        trainer: pl.Trainer,
        trainer_args: dict,
        tr_loader: torch.utils.data.DataLoader,
        vl_loader: torch.utils.data.DataLoader,
        monitor: str,
        max_: bool,
    ) -> float:
        """
        Trains the model and evaluates its performance.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to train.
            trainer (pl.Trainer): Trainer instance for managing training.
            tr_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            vl_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            trainer_args (dict): Trainer configuration.
            model_args (dict): Model parameters.
            monitor (str): Metric to evaluate.
            max_ (bool): Whether to maximize or minimize the metric.

        Returns:
            float: Evaluation metric for the model.
        """
        with open(
            f"{trainer_args["logger"].save_dir}/out.txt", "a"
        ) as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            trainer.fit(model, tr_loader, vl_loader)
            best_model = model_class.load_from_checkpoint(
                trainer_args["callbacks"][0].best_model_path, **model_args
            )
            results = trainer.validate(best_model, vl_loader)[0]

            with open(f"{trainer_args["logger"].save_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)

        if max_:
            return -results[monitor]

        return results[monitor]

    def constraints(self, params: list) -> bool:
        """
        Validates the given hyperparameters against the provided constraints.

        Args:
            params (list): List of hyperparameter values.

        Returns:
            bool: True if constraints are satisfied, otherwise False.
        """
        params = dict(zip([x.name for x in self.space], params))
        return self.constraint(params)

    def evaluate_params(self, params_list: list) -> list:
        """
        Evaluates multiple sets of hyperparameters in parallel.

        Args:
            params_list (list): List of hyperparameter sets to evaluate.

        Returns:
            list: Evaluation results for each set of hyperparameters.
        """
        results = Parallel(n_jobs=min(cpu_count(), len(params_list)))(
            delayed(self.objective)(params) for params in params_list
        )
        return results

    def __call__(self) -> list:
        """
        Executes the hyperparameter optimization process.

        Returns:
            list: Best hyperparameter set found during optimization.
        """
        n_parallel = min(cpu_count(), 3, self.n_calls)
        optimizer = Optimizer(
            dimensions=self.space,
            random_state=self.seed,
            space_constraint=self.constraints,
            n_initial_points=self.n_initial_points,
            n_jobs=n_parallel,
        )

        for _ in range(self.n_calls // n_parallel):
            params_list = optimizer.ask(n_points=n_parallel)
            losses = self.evaluate_params(params_list)
            optimizer.tell(params_list, losses)

        best_params = optimizer.Xi[np.argmin(optimizer.yi)]
        return best_params
