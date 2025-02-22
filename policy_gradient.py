import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
np.bool8 = np.bool_


# --------------------------------------------------
# Function to Generate Real Expert Data from CartPole
# --------------------------------------------------
def generate_expert_data(env, num_episodes=100, max_timesteps=500):
    """
    Runs an expert policy on the CartPole environment to collect
    state and expert action pairs.
    The expert policy here is a simple heuristic:
      if pole angle > 0 then action = 1, else action = 0.
    """
    states = []
    actions = []
    for episode in range(num_episodes):
        # Handle gym.reset() tuple (gym>=0.26) if needed
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        for t in range(max_timesteps):
            # Expert policy based on the pole angle (index 2 of state)
            action = 1 if state[2] > 0 else 0
            states.append(state)
            actions.append(action)
            next_state, reward, done, *_ = env.step(action)
            state = next_state
            if done:
                break
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    return states, actions


# --------------------------------------------------
# Core Modules: Actor, Critic, Memory
# --------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        logits = self.fc2(x)
        return logits


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        value = self.fc2(x)
        return value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# --------------------------------------------------
# Agents: PPOAgent and GRPOAgent
# --------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, beta=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.beta = beta  # coefficient for KL divergence penalty

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        # Copy of actor for computing importance sampling ratios
        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def select_action(self, state):
        # Unpack tuple from env.reset() if needed
        if isinstance(state, tuple):
            state = state[0]
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        logits = self.actor_old(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def evaluate(self, states, actions):
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states).squeeze()
        return action_logprobs, state_values, dist_entropy, logits

    def compute_kl(self, states):
        with torch.no_grad():
            old_logits = self.actor_old(states)
            old_probs = torch.softmax(old_logits, dim=-1)
        new_logits = self.actor(states)
        new_probs = torch.softmax(new_logits, dim=-1)
        kl = torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=-1)
        return kl.mean()

    def update(self, memory, K_epochs=4):
        states = torch.FloatTensor(np.array(memory.states))
        actions = torch.LongTensor(memory.actions)
        old_logprobs = torch.FloatTensor(memory.logprobs)

        discounted_rewards = []
        reward_sum = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                reward_sum = 0
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.insert(0, reward_sum)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        for _ in range(K_epochs):
            logprobs, state_values, dist_entropy, _ = self.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = nn.MSELoss()(state_values, discounted_rewards)
            kl_div = self.compute_kl(states)
            loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy.mean() + self.beta * kl_div

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())


class GRPOAgent(PPOAgent):
    """
    GRPOAgent extends PPOAgent by adding a gradient regularization term.
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 beta=0.01, lambda_grad=1e-3):
        super(GRPOAgent, self).__init__(state_dim, action_dim, lr, gamma, eps_clip, beta)
        self.lambda_grad = lambda_grad  # gradient penalty coefficient

    def compute_gradient_penalty(self, states, actions):
        states.requires_grad_(True)
        new_logits = self.actor(states)
        probs = torch.softmax(new_logits, dim=-1)
        dist = Categorical(probs)
        logp = dist.log_prob(actions)
        grad_outputs = torch.ones_like(logp)
        gradients = torch.autograd.grad(outputs=logp, inputs=states,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True,
                                        only_inputs=True)[0]
        grad_penalty = gradients.pow(2).sum(dim=1).mean()
        states.requires_grad_(False)
        return grad_penalty

    def update(self, memory, K_epochs=4):
        states = torch.FloatTensor(np.array(memory.states))
        actions = torch.LongTensor(memory.actions)
        old_logprobs = torch.FloatTensor(memory.logprobs)

        discounted_rewards = []
        reward_sum = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                reward_sum = 0
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.insert(0, reward_sum)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        for _ in range(K_epochs):
            logprobs, state_values, dist_entropy, _ = self.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = nn.MSELoss()(state_values, discounted_rewards)
            kl_div = self.compute_kl(states)
            grad_penalty = self.compute_gradient_penalty(states, actions)
            loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy.mean() \
                   + self.beta * kl_div + self.lambda_grad * grad_penalty

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())


# --------------------------------------------------
# BaseFinetuner: Modular Functions for Supervised & RL Finetuning
# --------------------------------------------------
class BaseFinetuner:
    def __init__(self, agent, supervised_data, batch_size=32, validation_split=0.2):
        """
        agent: instance of PPOAgent or GRPOAgent
        supervised_data: tuple (inputs, targets) as tensors for supervised training
        In this case, targets are the expert action labels.
        """
        self.agent = agent
        self.inputs, self.targets = supervised_data
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.train_loader = None
        self.val_loader = None
        self.supervised_loss_history = []
        self.rl_reward_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.actor.to(self.device)
        self.agent.critic.to(self.device)

    def prepare_dataloaders(self):
        """Prepare training and validation DataLoaders."""
        dataset = TensorDataset(self.inputs, self.targets)
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

    def extract_logits(self, x):
        """Extract logits from the agent's actor."""
        x = x.to(self.device)
        logits = self.agent.actor(x)
        return logits

    def get_probability_distribution(self, logits):
        """Return a probability distribution based on logits."""
        probs = torch.softmax(logits, dim=-1)
        return Categorical(probs)

    def default_reward_function(self, predictions, targets):
        """
        Compute a default reward based on negative cross entropy loss.
        Higher reward corresponds to lower error.
        """
        loss = nn.CrossEntropyLoss()(predictions, targets.to(predictions.device))
        return -loss

    def sample_group_info(self, data, group_size=10):
        """Dummy implementation for sampling group information."""
        indices = torch.randperm(len(data))[:group_size]
        return indices

    def compute_advantages(self, rewards, values):
        """Simple advantage: reward - value."""
        advantages = rewards - values.detach()
        return advantages

    def compute_surrogate_loss(self, ratios, advantages):
        """Compute the PPO surrogate loss with clipping."""
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.agent.eps_clip, 1 + self.agent.eps_clip) * advantages
        return -torch.min(surr1, surr2).mean()

    def compute_kl_divergence(self, states):
        """Compute KL divergence using the agent's method."""
        return self.agent.compute_kl(states)

    def rl_finetuning_generic(self, env_name='CartPole-v1', num_episodes=300, update_timestep=2000):
        """Perform RL fine-tuning using the agent's update method."""
        env = gym.make(env_name)
        memory = Memory()
        timestep = 0
        self.rl_reward_history = []

        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            ep_reward = 0
            done = False

            while not done:
                timestep += 1
                action, logprob, _ = self.agent.select_action(state)
                next_state, reward, done, *_ = env.step(action)

                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                state = next_state
                ep_reward += reward

                if timestep % update_timestep == 0:
                    self.agent.update(memory)
                    memory.clear()

            self.rl_reward_history.append(ep_reward)
            if episode % 10 == 0:
                avg_reward = np.mean(self.rl_reward_history[-10:])
                print(f"[RL Finetuning] Episode {episode} Average Reward: {avg_reward:.2f}")

    def supervised_training(self, num_epochs=20, learning_rate=1e-3):
        """Perform supervised training using real expert data."""
        self.prepare_dataloaders()
        optimizer = optim.Adam(self.agent.actor.parameters(), lr=learning_rate)
        # Use CrossEntropyLoss for classification (expert actions)
        loss_fn = nn.CrossEntropyLoss()
        self.supervised_loss_history = []

        for epoch in range(1, num_epochs + 1):
            epoch_losses = []
            for batch_inputs, batch_targets in self.train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                logits = self.agent.actor(batch_inputs)
                loss = loss_fn(logits, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            self.supervised_loss_history.append(avg_loss)
            print(f"[Supervised Training] Epoch {epoch}: Loss = {avg_loss:.4f}")

    def train_model(self, supervised_epochs=20, rl_episodes=300):
        """Train with supervised training first, then RL fine-tuning."""
        print("Starting supervised training...")
        self.supervised_training(num_epochs=supervised_epochs)
        print("Starting RL fine-tuning...")
        self.rl_finetuning_generic(num_episodes=rl_episodes)

    def evaluate_model(self):
        """Evaluate the model on the validation set using cross entropy loss."""
        self.agent.actor.eval()
        losses = []
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                logits = self.agent.actor(batch_inputs)
                loss = loss_fn(logits, batch_targets)
                losses.append(loss.item())
        self.agent.actor.train()
        avg_loss = np.mean(losses)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def supervised_training_plot(self):
        """Plot the supervised training loss curve with weighted moving average."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.supervised_loss_history, marker='o', label='Supervised Training Loss')
        if len(self.supervised_loss_history) >= 5:
            weighted_avg = np.convolve(self.supervised_loss_history, np.ones(5) / 5, mode='valid')
            plt.plot(np.arange(4, len(self.supervised_loss_history)), weighted_avg, color='red',
                     label='Weighted Avg (5 epochs)')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Supervised Training Loss")
        plt.legend()
        plt.show()
    def rl_finetuning_generic_plot(self):
        """Plot the RL fine-tuning reward curve with weighted moving average."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.rl_reward_history, marker='o', label='RL Finetuning Reward')
        if len(self.rl_reward_history) >= 10:
            weighted_avg = np.convolve(self.rl_reward_history, np.ones(10) / 10, mode='valid')
            # Adjust the x-axis indices for the moving average line
            plt.plot(np.arange(9, len(self.rl_reward_history)), weighted_avg, color='red',
                     label='Weighted Avg (10 episodes)')
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("RL Finetuning Reward")
        plt.legend()
        plt.show()

# --------------------------------------------------
# Agent-Specific Finetuners
# --------------------------------------------------
class PPOFinetuner(BaseFinetuner):
    """Finetuner for PPOAgent."""
    pass


class GRPOFinetuner(BaseFinetuner):
    """Finetuner for GRPOAgent."""
    pass


# --------------------------------------------------
# Example Usage: Training with PPO and GRPO using Real Expert Data
# --------------------------------------------------
if __name__ == '__main__':
    # Create the CartPole environment to get dimensions and generate expert data
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Generate real expert data from CartPole using our expert policy
    expert_states, expert_actions = generate_expert_data(env, num_episodes=100, max_timesteps=500)

    # ----------------------------
    # Example with PPOAgent
    # ----------------------------
    print("\n--- Training with PPOAgent ---")
    ppo_agent = PPOAgent(state_dim, action_dim)
    ppo_finetuner = PPOFinetuner(ppo_agent, (expert_states, expert_actions), batch_size=32)
    ppo_finetuner.train_model(supervised_epochs=10, rl_episodes=5000)
    ppo_finetuner.evaluate_model()
    ppo_finetuner.supervised_training_plot()
    ppo_finetuner.rl_finetuning_generic_plot()

    # ----------------------------
    # Example with GRPOAgent
    # ----------------------------
    print("\n--- Training with GRPOAgent ---")
    grpo_agent = GRPOAgent(state_dim, action_dim, lambda_grad=1e-3)
    grpo_finetuner = GRPOFinetuner(grpo_agent, (expert_states, expert_actions), batch_size=32)
    grpo_finetuner.train_model(supervised_epochs=10, rl_episodes=5000)
    grpo_finetuner.evaluate_model()
    grpo_finetuner.supervised_training_plot()
    grpo_finetuner.rl_finetuning_generic_plot()
