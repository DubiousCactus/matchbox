<div align="center">

# No-fuss-PyTorch-template

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra-zen](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://mit-ll-responsible-ai.github.io/hydra-zen/index.html)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
isort
autoflake
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

A clean template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this
template</kbd>](https://github.com/DubiousCactus/no-fuss-pytorch-template/generate) to initialize
new repository.

_Suggestions are always welcome!_

</div>

<br>



# Aim

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
