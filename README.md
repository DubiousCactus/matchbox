<div align="center">

# No fuss! PyTorch template

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra-zen](https://img.shields.io/badge/Config-Hydra--Zen-9cf)](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A minimalist PyTorch template with a terminal display that stays out of your way!

Click on [<kbd>Use this
template</kbd>](https://github.com/DubiousCactus/no-fuss-pytorch-template/generate) to initialize a
new repository.

_Suggestions are always welcome!_

</div>

<br>



# Aim

<div align="center">
<img src="https://www.scss.tcd.ie/~moralest/nfpt-display.gif">
Eyecandy display for your training curves in the terminal!
</div>
<br>

This template is intented to quickly bootstrap a PyTorch template with all the **necessary**
boilerplate that one typically writes in every project.

The ideas of this template are:
- Keep it DRY
    - Use hydra-zen to configure the experiments
- Raw PyTorch for maximum flexibility and transparency
- Minimal abstraction and opacity
- The sweet spot between a template and a framework
    - The bare minimum boilerplate is taken care of but not hidden away
    - The user is free to do whatever they want, everything is transparent
    - Provide base classes for datasets, models, etc. to make it easier to get started and provide
      good structure for DRY code and easy debugging
    - Provide a good set of defaults for the most common use cases
    - Provide a good set of tools to make it easier to debug and visualize
- Good Python practices enforced with git hooks:
    - Black
    - Isort
    - Autoflake
- Eyecandy progress bar and terminal plots of your training curves
	- tqdm
	- Rich
	- Plotext

### Why this one?

While the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) is clean,
mature, highly functional and comes with all the features you could ever want, I find that because
of all the abstraction it actually becomes a hindrance when writing research code. Unless you know
the template on the tip of your fingers, it's too much hassle to look for the right file, the right
method where you have to put your code. And PyTorch Lightning? Great for building stuff, not so
much for researching new Deep Learning architectures or working with custom data loading
procedures.

This template writes the **necessary boilerplate** for you, while **staying out of your
way**.


## Features

- **DRY configuration with Hydra-Zen**: painlessly configure experiments and swap whole groups with
    the CLI, simpler and DRYier than Hydra!
- **Run isolation and experiment reproducibility**: Hydra isolates each of your run and saves a
    YAML file of your config so you can always backtrack in your ML experiments.
- **Gorgeous terminal UI**: no more waiting for the slow Weights&Biases UI to load and sync,
    the curves are in your terminal! An informative and good-looking progress bar lets you know
    just what you need to know.
- **Weights & Biases integration**: of course, take advantage of WANDB when your code is ready to
    be launched into orbit.
- **Best-n model saver**: automatically deletes obsolete models from earlier epochs and only keeps
 the N best validation models on disk!
- **Automatic loading of the best model from a run name**: no need to look for that good model
    file, just pass the run name that was generated for you!
- **SIGINT handler**: waits for the end of the epoch and validation to terminate the programm if
     CTRL+C is pressed.


## Structure

```
my-pytorch-project/
    conf/
        experiment.py <-- experiment-level configurations
        project.py <-- project-level constants
    data/
        . <-- your dataset goes here
    dataset/
        base/
            __init__.py <-- base dataset implementation
            image.py <- base image dataset implementation
        . <-- your dataset implementation goes here
    model/
        . <-- your model implementation goes here
    scripts/
        resize_image_dataset.py
        . <-- your utility scripts go here
    src/
        base_trainer.py <-- the core of the training logic
    utils/
        __init__.py <-- low-level utilities
        helpers.py <-- high-level utilities
        training.py <-- training related utilities
    vendor/
        . <-- third-party code goes here
    train.py <-- training entry point
    test.py <-- testing entry point
```

## Core logic

The core of this template is implemented in `src/base_trainer.py`:
```python
class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        opt: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Base trainer class.
        Args:
            model (torch.nn.Module): Model to train.
            opt (torch.optim.Optimizer): Optimizer to use.
            train_loader (torch.utils.data.DataLoader): Training dataloader.
            val_loader (torch.utils.data.DataLoader): Validation dataloader.
        """

    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> torch.Tensor:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        # TODO: Your code goes here!

    def _train_epoch(self, description: str, epoch: int) -> float:
        """Perform a single training epoch.
        Args:
            description (str): Description of the epoch for tqdm.
            epoch (int): Current epoch number.
        Returns:
            float: Average training loss for the epoch.
        """

    def _val_epoch(self, description: str, visualize: bool, epoch: int) -> float:
        """Validation loop for one epoch.
        Args:
            description: Description of the epoch for tqdm.
            visualize: Whether to visualize the model predictions.
        Returns:
            float: Average validation loss for the epoch.
        """
        "==================== Validation loop for one epoch ===================="

    def train(
        self,
        epochs: int = 10,
        val_every: int = 1,  # Validate every n epochs
        visualize_every: int = 10,  # Visualize every n validations
    ):
        """Train the model for a given number of epochs.
        Args:
            epochs (int): Number of epochs to train for.
            val_every (int): Validate every n epochs.
            visualize_every (int): Visualize every n validations.
        Returns:
            None
        """

    def _setup_plot(self):
        """Setup the plot for training and validation losses."""

    def _plot(self, epoch: int, train_losses: List[float], val_losses: List[float]):
        """Plot the training and validation losses.
        Args:
            epoch (int): Current epoch number.
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.
        Returns:
            None
        """

    def _save_checkpoint(self, val_loss: float, ckpt_path: str, **kwargs) -> None:
        """Saves the model and optimizer state to a checkpoint file.
        Args:
            val_loss (float): The validation loss of the model.
            ckpt_path (str): The path to the checkpoint file.
            **kwargs: Additional dictionary to save. Use the format {"key": state_dict}.
        Returns:
            None
        """

    def _load_checkpoint(self, ckpt_path: str) -> None:
        """Loads the model and optimizer state from a checkpoint file.
        Args:
            ckpt_path (str): The path to the checkpoint file.
        Returns:
            None
        """

    def _terminator(self, sig, frame):
        """
        Handles the SIGINT signal (Ctrl+C) and stops the training loop.
        """
```

## Setting up

Run `pre-commit install` to setup the pre-commit hooks. These will run [Black](), [Isort](),
[Autoflake]() and others to clean up your code.


## Roadmap
 - [x] Torchmetrics: this takes care of batching the loss and averaging it
 - [x] Saving top 3 best val models (easily configure metric)
 - [x] Training + evaluation loop
 - [x] Wandb integration with predefined logging metrics
 - [x] Automatic instantiation for the optimizer, scheduler, model
 - [x] The best progress display I can ever get!! (kinda like torchlightning template? But I want
 colour (as in PIP), I want to see my hydra conf, and I want to see a little graph in a curses style in real
 time (look into Rich, Textual, etc.).
 - [x] Interception of SIGKILL, SIGTERM to stop training but save everything: two behaviours (1
 will be the default ofc) -- a) wait for epoch end and validation, b) abort epoch.
 - [x] Add git hooks for linting, formatting, etc.
 - [x] Training logic
 - [ ] Test logic
 - [ ] Generate random run name (human readble from words)
 - [ ] Automatic model loading from run name
