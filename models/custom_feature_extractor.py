"Some custom feature extractor for sb3."
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer

from torch_geometric.nn.models import GAT
from torch_geometric.data import Batch
from torch_geometric.data import Data


class CustomGAT(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param encoder_hidden_dim: (int) Hidden dimension in each encoder layer.
    :param encoder_out_dim: (int) Dimension of the output from the encoder.
    :param encoder_num_layers: (int) Number of layers in the encoder.
    :param gcn_num_layers: (int) Number of layers in the GCN.

    Note: The total number of features extracted remains fixed at 16.
    """

    def __init__(self,
                 observation_space: spaces.Dict,
                 encoder_hidden_dim: int = 32,
                 encoder_out_dim: int = 16,
                 encoder_num_layers: int = 2,
                 gnn_num_layers: int = 1,
                 gnn_hidden_channels: int = 16,
                 gnn_out_channels: int = 16):
        super().__init__(observation_space, gnn_out_channels)

        n_input = observation_space['node_feat'].shape[-1]

        # Create a list of encoder layers based on the specified number of layers
        encoder_layers = []
        for i in range(encoder_num_layers):
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(n_input if i == 0 else encoder_hidden_dim,
                              encoder_hidden_dim), nn.ReLU()))
        encoder_layers.append(nn.Linear(encoder_hidden_dim, encoder_out_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        self.gnn = GAT(in_channels=n_input,
                       hidden_channels=gnn_hidden_channels,
                       num_layers=gnn_num_layers,
                       out_channels=gnn_out_channels)

    def forward(self, obs) -> th.Tensor:
        graphs = []
        for i in range(obs['node_feat'].shape[0]):
            graph = Data(x=obs['node_feat'][i],
                         edge_index=obs['adjacency'][i].nonzero().t(),
                         mask=obs['mask'][i])
            graphs.append(graph)
        graphs = Batch.from_data_list(graphs)
        for layer in self.encoder:
            graphs.x = layer(graphs.x)
        h2 = self.gnn(graphs.x, graphs.edge_index)
        return h2[graphs.mask == 1]


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels,
                      32,
                      kernel_size=[8, 3],
                      stride=4,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=[4, 1], stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

