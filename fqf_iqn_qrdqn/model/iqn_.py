import torch

from .base_model import BaseModel
from fqf_iqn_qrdqn.network_ import DQNBase, CosineEmbeddingNetwork,\
    QuantileNetwork


class IQN(BaseModel):

    def __init__(self, state_dim, num_actions, K=32, num_cosines=32,
                  dueling_net=False, noisy_net=False):
        super(IQN, self).__init__()
        # Feature extractor of DQN.
        # self.dqn_net = DQNBase(num_channels=num_channels)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, state_dim=state_dim,
            noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net)

        self.K = K
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def calculate_state_embeddings(self, states):
        return states.reshape(states.shape[0], -1)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = states.reshape(states.shape[0], -1)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
