<div align="center">

# Bells & Whistles: a PyTorch template

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra-zen](https://img.shields.io/badge/Config-Hydra--Zen-9cf)](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A batteries-included PyTorch template with a terminal display that stays out of your way!

Click on [<kbd>Use this
template</kbd>](https://github.com/DubiousCactus/bells-and-whistles/generate) to initialize a
new repository.

_Suggestions are always welcome!_

</div>

<br>



# Why use this template?

<div align="center">
<img src="https://www.scss.tcd.ie/~moralest/nfpt-display.gif">
<i>Eyecandy display for your training curves in the terminal!</i>
</div>
<br>

It is intented to quickly bootstrap a PyTorch project with all the **necessary**
boilerplate that one typically writes in every project. We give you all the *bells and whistles* so
you can focus on what matters.

<h3><i>Key ideas:</i></h3>

- Keep it DRY (Don't Repeat Yourself) and repeatable with
	[Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html).
- Raw PyTorch for maximum flexibility and transparency.
- Minimal abstraction and opacity but focus on extensibility.
- The sweet spot between a template and a framework.
    - The bare minimum boilerplate is taken care of but not hidden away.
    - You are free to do whatever you want, everything is transparent.
	- Provides base classes for datasets, trainer, etc. to make it easier to get started and provide
      good structure for DRY code and easy debugging.
    - Provides a set of defaults for the most common use cases.
    - Provides a set of tools to make it easier to debug and visualize.
	- Provides all the [bells and whistles](https://github.com/DubiousCactus/bells-and-whistles) to
		make your programming experience *fun*!

### Why this one?

While the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) is clean,
mature, highly functional and comes with all the features you could ever want, I find that because
of all the abstraction it actually becomes a hindrance when writing research code. Unless you know
the template on the tip of your fingers, it's too much hassle to look for the right file, the right
method where you have to put your code. And PyTorch Lightning? Great for building stuff, not so
much for researching new training pipelines or working with custom data loading.

<div align="center">
<h4>
This template writes the necessary boilerplate for you, while staying out of your way.
</h4>
</div>


## Features

- **DRY configuration with [Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)**: painlessly configure experiments and swap whole groups with
    the CLI, simpler and DRYier than Hydra!
- **Run isolation and experiment reproducibility**: Hydra isolates each of your run and saves a
    YAML file of your config so you can always backtrack in your ML experiments.
- **Gorgeous terminal UI**: no more waiting for the slow Weights&Biases UI to load and sync,
    the curves are in your terminal (thanks to [Plotext](https://github.com/piccolomo/plotext/))! An informative and good-looking progress bar lets you know
    just what you need to know.
- **[Weights & Biases](https://wandb.ai) integration**: of course, take advantage of WANDB when your code is ready to
    be launched into orbit.
- **Best-n model saver**: automatically deletes obsolete models from earlier epochs and only keeps
 the N best validation models on disk!
- **Automatic loading of the best model from a run name**: no need to look for that good model
    file, just pass the run name that was generated for you. They have cool and colorful names :)
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

## Setting up

1. Set up a virtual environment and activate it.
2. [Install PyTorch](https://pytorch.org/get-started/) for your system.
3. Run `pip install -r requirements.txt`.
4. Run `pre-commit install` to setup the pre-commit hooks. These will run [Black](), [Isort](),
[Autoflake]() and others to clean up your code.


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
 - [x] Generate random run name (human readble from words)
 - [x] Automatic model loading from run name
 - [x] Test logic
 - [x] requirements.txt
 - [ ] Tests?
 - [ ] Feedback & improvements
