#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for the experiments, using hydra-zen.
"""

import torch
from hydra_zen import builds, make_config, make_custom_builds_fn, store

from dataset.base.image import ImageDataset


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)

MyModelConf = builds(MyModel)
model_a = MyModelConf(
    encoder_layers=4,
    decoder_layers=3,
)
model_b = MyModelConf(
    encoder_layers=14,
    decoder_layers=13,
)
model_store = store(group="exp/model")
model_store(model_a, name="model_a")
model_store(model_b, name="model_b")

MyDatasetConf = pbuilds(
    ImageDataset,
    dataset_root="path/to/dataset",
    enable_augs=True,
    normalize=False,
)

adam = pbuilds(torch.optim.Adam, lr=1e-3)
sdg = pbuilds(torch.optim.SGD, lr=1e-3)

opt_store = store(group="exp/optimizer")
opt_store(adam, name="adam")
opt_store(sdg, name="sdg")

DataLoaderConf = pbuilds(
    torch.utils.data.DataLoader,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
)

# So this will be the default experiment config
BaseExperiementConfig = make_config(
    seed=1,
    opt=adam,
    train_dataset=pbuilds(ImageDataset, builds_bases=(MyDatasetConf,), split="train"),
    train_loader=pbuilds(
        torch.utils.data.DataLoader, builds_bases=(DataLoaderConf,), shuffle=True
    ),
    val_dataset=pbuilds(ImageDataset, builds_bases=(MyDatasetConf,), split="val"),
    val_loader=pbuilds(
        torch.utils.data.DataLoader, builds_bases=(DataLoaderConf,), shuffle=False
    ),
    epochs=100,
    val_every=1,
    visualize_every=0,
    tiny_dataset=False,
    model=None,
)

ModelAExperimentConfig = make_config(
    bases=(BaseExperiementConfig,),
    model=model_a,
    tiny_dataset=True,
)

ModelBExperimentConfig = make_config(
    bases=(BaseExperiementConfig,),
    model=model_b,
    epochs=200,
    val_every=10,
)


experiment_store = store(group="exp")
experiment_store(BaseExperiementConfig, name="base")
experiment_store(ModelAExperimentConfig, name="model_a")
experiment_store(ModelBExperimentConfig, name="model_b")
