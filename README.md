<div align="center">

# Matchbox

[![Run & validate](https://github.com/DubiousCactus/bells-and-whistles/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DubiousCactus/bells-and-whistles/actions/workflows/python-app.yml)
[![python](https://img.shields.io/badge/-Python_3.10_--%3E_3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra-zen](https://img.shields.io/badge/Config-Hydra--Zen-9cf)](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Vim](https://img.shields.io/badge/VIM%20ready!-forestgreen?style=for-the-badge&logo=vim)](https://github.com/DubiousCactus/bells-and-whistles/blob/main/.vimspector.json)

A transparent PyTorch micro-framework for pragmatic research code.
<!-- A batteries-included PyTorch template with a terminal display that stays out of your way! -->

Click on [<kbd>Use this
template</kbd>](https://github.com/DubiousCactus/bells-and-whistles/generate) to initialize a
new repository.

_Suggestions are always welcome!_

</div>

<br>


<div align="center">
<img src="https://www.scss.tcd.ie/~moralest/nfpt-display.gif">  <!-- TODO: update -->
<i>Eyecandy display for your training curves in the terminal!</i>
</div>
<br>



# What comes with Matchbox?

This framework is intended to quickly bootstrap a PyTorch project with all the
**necessary** boilerplate that one typically writes in every project. We give you all
the bells and whistles, so you can focus on what matters.

Yes, we love [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) too,
but we feel that it's not flexible enough for research code and fast iteration. Too much
opacity and abstraction sometimes kills productivity.

Matchbox comes with **3 killer features**.

### 1. A Text User Interface (TUI)

A useful Text User Interface (TUI) for instant access to training loss curves. No more
loading heavy web apps and waiting for synchronization; you don't need to leave the
terminal (even through SSH sessions)!

Of course, [Weights & Biases](https://wandb.ai) is still integrated in Matchbox ;)

<details>
    <summary>Video demo</summary> <!-- TODO: -->
</details>

### 2. A pragmatic PyTorch micro-framework

No more failed experiments due to stale dataset caches, no more spending hours
recomputing a dataset because implementing multiprocessing would require a massive
refactoring, and no more countless scrolls to find the parameters of your model after
training.

Matchbox gives you a framework that you can jump right in. It boils down to:
- A minimal PyTorch project template.
- A bootstrapping boilerplate for datasets, models, training & testing logic with
    powerful configuration via
    [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html).
- Dataset composable [mixins]() for pragmatic dataset implementation:
    - Safeguards for automatic and smart caching of data pre- and post-processing. Did
    you change a parameter? Modify a line of code? No worries, Matchbox will catch that
    and flush your cache.
    - Dataset pre-processing boilerplate: never write the multiprocessing code ever again,
        just write processing code *per-sample*!
- A set of utility functions for a broad range of deep learning projects.


### 3. An interactive coding experience for fast iteration


Matchbox fully erases the most painful part of writing deep learning research code:
no more relaunching the whole code to fix some tensor shapes that occur at runtime.
Matchbox will hold the expensive “cold code” in memory and let you work on the
quick-to-reload “hot code” via hot reloading.


This builder feature allows PyTorch developers to:

- Freeze entire components (dataset, model, training logic, etc.) in memory to work on the next feature with no time-waisting edit-reload-wait cycle!
- Experiment very quickly with hot code reloading!
- Catch exceptions and interactively debug tensor operations on their actual data!

And all of this graphically :)
<details>
    <summary>Video demo</summary> <!-- TODO: -->
</details>

<h3><i>Core principles of Matchbox:</i></h3>

- Keep it DRY (Don't Repeat Yourself) and repeatable with
	[Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html).
- Raw PyTorch for maximum flexibility and transparency.
- Minimal abstraction and opacity but focus on extensibility.
- The sweet spot between a template and a framework.
    - The bare minimum boilerplate is taken care of but not hidden away.
    - You are free to do whatever you want, everything is transparent: no pip package!
    - Provides base classes for datasets, trainer, etc. to make it easier to get started and provide
      good structure for DRY code and easy debugging.
    - Provides a set of defaults for the most common use cases.
    - Provides a set of tools to make it easier to debug and visualize.
    - Provides all the bells and whistles to make your programming experience *fun*!

# Why chose Matchbox over the alternatives?

While the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) is clean,
mature, highly functional and comes with all the features you could ever want, we find that because
of all the abstraction it actually becomes a hindrance when writing research code. Unless you know
the template on the tip of your fingers, it's too much hassle to look for the right file, the right
method where you have to put your code. And [PyTorch
Lightning](https://github.com/Lightning-AI/pytorch-lightning)? Great for building stuff,
not so much for researching new training pipelines or working with custom data loading.

<div align="center">
<h5>
This template writes the necessary boilerplate for you, while staying out of your way.
</h5>
</div>


## Features

- **DRY configuration with
    [Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)**: painlessly
    configure experiments and swap whole groups with the CLI, simpler and DRYier than
    Hydra!
- **Run isolation and experiment reproducibility**: Hydra isolates each of your run and
    saves a YAML file of your config so you can always backtrack in your ML experiments.
- **Gorgeous terminal UI**: no more waiting for the slow Weights&Biases UI to load and
    sync, the curves are in your terminal (thanks to
    [Plotext](https://github.com/piccolomo/plotext/) and
    [Textual](https://github.com/Textualize/textual))! An informative and good-looking TUI
    lets you know just what you need to know.
- **[Weights & Biases](https://wandb.ai) integration**: of course, take advantage of W&B
    when your code is ready to be launched into orbit.
- **Best-n model saver**: automatically deletes obsolete models from earlier epochs and
    only keeps the N best validation models on disk!
- **Automatic loading of the best model from a run name**: no need to look for that good model file, just pass the run name that was generated for you. They have cool and
    colorful names :)
<!-- - **SIGINT handler**: waits for the end of the epoch and validation to terminate the -->
<!--     programm if CTRL+C is pressed. -->


# Getting started

Have a look at our [documentation](). If you are still struggling, open a discussion
thread on this repo and we'll help you ASAP :)

## Structure

```
my-pytorch-project/
    bootstrap/
        factories.py <-- Factory functions for instantiating models, optimizers, etc.
        launch_experiment.py <-- Bootstraps the experiment and launches the training/testing loop
    conf/
        experiment.py <-- experiment-level configurations
        project.py <-- project-level constants
    data/
        . <-- your dataset files and cache/preprocessing output go here
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
        losses/
            . <-- Custom losses go here
        metrics/
            . <-- Custom metrics go here
        base_tester.py <-- the core of the testing logic
        base_trainer.py <-- the core of the training logic
    utils/
        __init__.py <-- low-level utilities
        helpers.py <-- high-level utilities
        training.py <-- training-related utilities
    vendor/
        . <-- third-party code goes here (github submodules, etc.)
    train.py <-- training entry point (calls bootstrap/launch_experiment)
    test.py <-- testing entry point (calls bootstrap/launch_experiment)
```

## Setting up

1. Set up a virtual environment and activate it.
2. [Install PyTorch](https://pytorch.org/get-started/) for your system.
3. Run `pip install -r requirements.txt`.
4. Run `pre-commit install` to setup the pre-commit hooks. These will run [Black](), [Isort](),
[Autoflake]() and others to clean up your code before each commit.

## Usage

A typical way of using this template is to follow these steps:

1. Implement your dataset loader (look at `dataset/example.py`)
2. Configure it (look at the `dataset` section in `conf/experiment.py`)
3. Implement your model (look at `model/example.py`)
4. Configure it (look at the `model` section in `conf/experiment.py`)
5. Configure your entire experiment(s) in `conf/experiment.py`.
6. Implement `_train_val_iteration()` in `src/base_trainer.py`, or derive your own
   trainer for more complex use cases.

To run an experiment, use `./train.py +experiment=my_experiment`.

You may experiment on the fly with `./train.py +experiment=my_experiment
dataset=my_dataset data_loader.batch_size=32 model.latent_dim=128 run.epochs=30
run.viz_every=5`.

To evaluate a model, run `test.py +experiment=my_experiment
run.load_from_run=<run_name_from_previous_training>`.


You can always look at what's available in your config with `./train.py --help` or
`./test.py --help`!


### Configuring your experiments & project

This template comes with an example dataset and model that are pre-configured in
`conf/experiment.py` using
[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html). To learn how to adapt it
to your project, you may refer to their
[documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/api_reference.html) and
[tutorials](https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials.html).

In addition, you can configure the behaviour of the training logic as well as other settings by
setting project-level constants in `conf/project.py`, such as the name of your project (i.e. for
[wandb.ai](https://wandb.ai)).

<details><summary>You can override most project-level constants at runtime using environment variables!</summary>

```
$ PLOT_ENABLED=false REPRODUCIBLE=0 ./train.py +experiment=my_experiment
```

</p>
</details>


### Logging and repeatability

Each of your run creates a folder with its name in `runs/`. You can find there the used YAML
config, the checkpoints and any other file you wish to save.

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
 - [ ] Feedback & improvements (continuous so don't expect this to ever be checked!)
 - [x] Refactor what is necessary (UI stuff, training & testing)
 - [ ] Tests?
 - [ ] Streamline the configuration (make it more DRY with either conf gen or runtime conf inference)
 - [ ] Make datasets highly reproducible to the max (masterplan):
	 - [x] Hash the dataset post-instantiation (iterate and hash) and log to wandb.
	 - [ ] Log the date of creation of all files (log anomalies like one date sticking out)
	 - [ ] ???
