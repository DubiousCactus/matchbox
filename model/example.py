#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example model.
"""

from torch import nn


class ExampleModel(nn.Module):
    def __init__(
        self,
        encoder_input_dim: int,
        encoder_dim: int,
        latent_dim: int,
        decoder_output_dim: int,
        decoder_dim: int,
    ) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, latent_dim),
        )
        # raise Exception("This is an exception")
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_output_dim),
        )

    def forward(self, x):
        x = self._encoder(x.view(x.size(0), -1))
        x = self._decoder(x)
        return x
