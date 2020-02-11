import torch
from torch.optim import Adam

from fqf_iqn.model import IQN
from fqf_iqn.utils import update_params, calculate_quantile_huber_loss
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=64, N_dash=64, K=32, num_cosines=64,
                 kappa=1.0, lr=5e-5, memory_size=10**6, gamma=0.99,
                 multi_step=1, update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 log_interval=50, eval_interval=250000, num_eval_steps=125000,
                 cuda=True, seed=0):
        super(IQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, log_interval,
            eval_interval, num_eval_steps, cuda, seed)

        # Implicit Quantile Networks.
        self.iqn = IQN(
            num_channels=self.env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, N_dash=N_dash, K=K,
            num_cosines=num_cosines, device=self.device)

        self.optim = Adam(
            list(self.iqn.dqn_base.parameters())
            + list(self.iqn.quantile_net.parameters()),
            lr=lr, eps=1e-2/batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            state_embedding = self.iqn.dqn_base(state)
            action = self.iqn.calculate_q(
                state_embedding).argmax().item()
        return action

    def learn(self):
        self.learning_steps += 1

        if self.steps % self.target_update_interval == 0:
            self.iqn.update_target()

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)
        loss = self.calculate_loss(
            states, actions, rewards, next_states, dones)
        update_params(self.optim, loss)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/loss', loss.detach().item(), self.learning_steps)

    def calculate_loss(self, states, actions, rewards, next_states, dones):

        # Calculate features of states.
        state_embeddings = self.iqn.dqn_base(states)

        # Sample fractions.
        taus = torch.rand(
            self.batch_size, self.N, dtype=states.dtype, device=states.device)

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.iqn.quantile_net(state_embeddings, taus)

        # Repeat current actions into (batch_size, N, 1).
        action_index = actions[..., None].expand(
            self.batch_size, self.N, 1)

        # Calculate quantile values of current states and current actions.
        current_sa_quantiles = current_s_quantiles.gather(
            dim=2, index=action_index).view(self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate features of next states.
            next_state_embeddings = self.iqn.dqn_base(next_states)

            # Sample next fractions.
            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=states.dtype,
                device=states.device)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.iqn.target_net(
                next_state_embeddings, tau_dashes)

            # Calculate next greedy actions.
            next_actions = torch.argmax(
                self.iqn.calculate_q(next_state_embeddings), dim=1
                ).view(self.batch_size, 1, 1)

            # Repeat next actions into (batch_size, num_taus, 1).
            next_action_index = next_actions.expand(
                self.batch_size, self.N_dash, 1)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = next_s_quantiles.gather(
                dim=2, index=next_action_index).view(-1, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.N_dash)

        # TD errors.
        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        # Calculate quantile huber loss.
        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, taus, self.kappa)

        return quantile_huber_loss

    def save_models(self):
        self.iqn.save(self.model_dir)