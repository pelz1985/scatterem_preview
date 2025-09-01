from typing import Any, Dict

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, network_config: Dict[str, Any], in_dims: int, out_dims: int
    ) -> None:
        super(MLP, self).__init__()
        self.in_dims: int = in_dims
        self.out_dims: int = out_dims
        self.activation: str = network_config["activation"]
        self.output_activation: str = network_config["output_activation"]
        self.n_neurons: int = network_config["n_neurons"]
        self.n_hidden_layers: int = network_config["n_hidden_layers"]

        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dims, self.n_neurons)]
            + [
                nn.Linear(self.n_neurons, self.n_neurons)
                for i in range(1, self.n_hidden_layers - 1)
            ]
        )
        self.layers.append(nn.Linear(self.n_neurons, self.out_dims))

        # Define the number of layers for activation assignment
        num_layers = len(self.layers)

        # Activation for hidden layers
        if self.activation == "ReLU":
            self.activations = nn.ModuleList([nn.ReLU() for i in range(num_layers - 1)])
        elif self.activation == "LeakyReLU":
            self.activations = nn.ModuleList(
                [nn.LeakyReLU() for i in range(num_layers - 1)]
            )
        elif self.activation == "Sigmoid":
            self.activations = nn.ModuleList(
                [nn.Sigmoid() for i in range(num_layers - 1)]
            )
        else:
            raise NotImplementedError(
                "Unknown activation, only support ReLU, LeakyReLU and Sigmoid"
            )

        # Activation for the last layer
        if self.output_activation == "ReLU":
            self.activations.append(nn.ReLU())
        elif self.output_activation == "LeakyReLU":
            self.activations.append(nn.LeakyReLU())
        elif self.output_activation == "Sigmoid":
            self.activations.append(nn.Sigmoid())
        elif self.output_activation == "None":
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x
