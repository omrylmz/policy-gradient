import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from typing import Any, Dict, List, Tuple
import numpy as np
np.bool8 = np.bool_


#########################
# 1. Data Preparation   #
#########################

def prepare_dataloaders(dataset, batch_size: int = 32):
    """
    Prepare DataLoader objects (or similar) for supervised training or RL usage.
    In RL, you might not always use a standard DataLoader, but this function
    can unify how you batch data from rollouts, replay buffers, or offline datasets.

    :param dataset: Typically a PyTorch Dataset or custom structure with experiences.
    :param batch_size: Batch size for supervised or RL data.
    :return: A PyTorch DataLoader (or a custom generator).
    """
    # Example: if it's a standard supervised dataset
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


#########################
# 2. Policy & Model     #
#########################

class SimplePolicy(nn.Module):
    """
    Example policy for a small RL environment like CartPole.
    For LLM training, you would replace this with a much larger model
    (e.g., a transformer-based architecture).
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        """
        Returns raw logits for the actions.
        """
        return self.net(x)


def extract_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Forward pass to get the unnormalized logits from the policy model.

    :param model: A PyTorch nn.Module representing the policy.
    :param inputs: Tensor representing observations (e.g., states).
    :return: Logits for each possible action.
    """
    return model(inputs)


def get_probability_distribution(logits: torch.Tensor):
    """
    Convert logits into a probability distribution object.
    This is typically used in sampling or computing log_prob.

    :param logits: Unnormalized log probabilities (N x action_dim).
    :return: A torch.distributions.Distribution object, e.g. Categorical.
    """
    return Categorical(logits=logits)


#########################
# 3. Reward & Advantage #
#########################

def default_reward_function(observations, actions, next_observations, done_flags) -> torch.Tensor:
    """
    Computes default rewards for the environment step.
    For CartPole, the reward is typically +1 per step until done.
    For LLM tasks, you would implement a custom reward (e.g., from a reward model).

    :param observations: Current observations (states).
    :param actions: Actions taken.
    :param next_observations: Next observations (states).
    :param done_flags: Boolean or 0/1 flags for terminal states.
    :return: A torch.Tensor of shape (batch_size,) with scalar rewards.
    """
    # Example for CartPole: reward = 1.0 each step
    return torch.ones(len(done_flags))


def sample_group_info(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    For RL tasks, gather group information from a rollout or replay buffer.
    For LLM tasks, it might be contextual info on prompts, etc.

    :param batch_data: Contains states, actions, rewards, next_states, done_flags, etc.
    :return: A dictionary with grouped info (e.g., timesteps, advantages, or anything needed).
    """
    # For simplicity, we'll just return the data as-is or do minimal grouping.
    return batch_data


def compute_advantages(rewards: torch.Tensor,
                       values: torch.Tensor,
                       next_values: torch.Tensor,
                       dones: torch.Tensor,
                       gamma: float = 0.99,
                       lam: float = 0.95) -> torch.Tensor:
    """
    Generalized Advantage Estimation (GAE) computation.

    :param rewards: Rewards at each timestep (T).
    :param values: Value function predictions at each timestep (T).
    :param next_values: Value function predictions at next timestep (T), or 0 for terminal.
    :param dones: Boolean flags indicating if episode is done at each timestep.
    :param gamma: Discount factor.
    :param lam: Lambda factor for GAE.
    :return: A torch.Tensor of shape (T,) with advantage estimates.
    """
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    return torch.stack(advantages)


#########################
# 4. Loss & Metrics     #
#########################

def compute_surrogate_loss(prob_dist_old, prob_dist_new, actions, advantages):
    """
    Computes the basic surrogate (policy gradient) loss for GRPO/PPO-like methods.

    :param prob_dist_old: The old policy distribution (from previous step).
    :param prob_dist_new: The new policy distribution (current step).
    :param actions: Actions taken (used to index log_prob).
    :param advantages: Advantage estimates for each action.
    :return: A torch scalar (the negative surrogate objective).
    """
    # Log probabilities under old and new policy
    log_probs_old = prob_dist_old.log_prob(actions)
    log_probs_new = prob_dist_new.log_prob(actions)
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Basic surrogate objective:
    # L = - E[ ratio * advantage ]
    # (In PPO, you'd clamp ratio, but for GRPO, we might keep it simpler.)
    surrogate = -torch.mean(ratio * advantages)
    return surrogate


def compute_kl_divergence(prob_dist_old, prob_dist_new):
    """
    Computes KL divergence between old and new distribution.
    For Categorical, we can do an analytical approach, or
    we can do a sampling-based approximation.

    :param prob_dist_old: Old policy distribution object.
    :param prob_dist_new: New policy distribution object.
    :return: KL divergence (scalar).
    """
    # The standard formula for KL(Cat(p) || Cat(q)) = sum( p * log(p/q) )
    p = prob_dist_old.probs
    q = prob_dist_new.probs
    kl_div = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1)
    return kl_div.mean()


#########################
# 5. Training Loops     #
#########################

def grpo_rl_finetuning_generic(
    env,
    policy_model,
    value_model,
    optimizer_policy,
    optimizer_value,
    epochs: int = 100,
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    lam: float = 0.95
):
    """
    RL finetuning loop for GRPO that:
      - Uses *separate* forward passes for policy & value (avoiding double-backward errors).
      - Converts list-of-ndarrays into np.array() for better performance.

    :param env: The environment (gym or gymnasium).
    :param policy_model: nn.Module for the policy (outputs logits).
    :param value_model: nn.Module for the value function (outputs scalar value).
    :param optimizer_policy: Optimizer for the policy model.
    :param optimizer_value: Optimizer for the value model.
    :param epochs: Number of update epochs.
    :param rollout_steps: Number of timesteps collected each epoch.
    :param gamma: Discount factor.
    :param lam: Lambda for GAE advantage estimation.
    :return: A dictionary (history) with training stats.
    """

    def get_probability_distribution(logits: torch.Tensor):
        return Categorical(logits=logits)

    def compute_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def compute_surrogate_loss(dist_old, dist_new, actions, advantages):
        logp_old = dist_old.log_prob(actions)
        logp_new = dist_new.log_prob(actions)
        ratio = torch.exp(logp_new - logp_old)
        # GRPO or PPO-style objective:
        return -torch.mean(ratio * advantages)

    def compute_kl_divergence(dist_old, dist_new):
        p = dist_old.probs
        q = dist_new.probs
        kl = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1)
        return kl.mean()

    history = {
        "episode_rewards": [],
        "kl_divergences": [],
    }

    for epoch in range(epochs):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones_list = []

        state, info = env.reset()  # Gymnasium: (obs, info)
        ep_reward = 0

        for _ in range(rollout_steps):
            state_t = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0)

            # --- POLICY FORWARD (old) ---
            with torch.no_grad():
                logits_old = policy_model(state_t)
                dist_old = get_probability_distribution(logits_old)
                action = dist_old.sample()

            # Step environment (Gymnasium: returns 5 values)
            next_state, reward, done, truncated, info = env.step(action.item())
            done = done or truncated

            # Store transition
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            next_states.append(next_state)
            dones_list.append(float(done))

            state = next_state
            ep_reward += reward

            if done:
                history["episode_rewards"].append(ep_reward)
                state, info = env.reset()
                ep_reward = 0

        # Convert to Tensors
        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64))
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32))
        next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones_t = torch.from_numpy(np.array(dones_list, dtype=np.float32))

        # --- VALUE FORWARD for next_states ---
        with torch.no_grad():
            next_values_t = value_model(next_states_t).squeeze(-1)

        # --- VALUE FORWARD for states ---
        values_t = value_model(states_t).squeeze(-1)

        # --- Compute advantages (GAE) ---
        advantages_t = compute_advantages(rewards_t, values_t, next_values_t, dones_t, gamma=gamma, lam=lam)

        # --- Old policy distribution (no grad) ---
        with torch.no_grad():
            logits_old_all = policy_model(states_t)
            dist_old_all = get_probability_distribution(logits_old_all)

        # --- Policy update (New forward pass) ---
        logits_new = policy_model(states_t)
        dist_new = get_probability_distribution(logits_new)

        policy_loss = compute_surrogate_loss(dist_old_all, dist_new, actions_t, advantages_t)

        optimizer_policy.zero_grad()
        policy_loss.backward()  # frees policy graph
        optimizer_policy.step()

        # --- Value update (Separate forward pass) ---
        # (We already have values_t = value_model(states_t) from above,
        #  but do it again if needed to keep it conceptually separate:
        # values_t = value_model(states_t).squeeze(-1))

        target_values = rewards_t + gamma * next_values_t * (1 - dones_t)
        value_loss = nn.MSELoss()(values_t, target_values.detach())

        optimizer_value.zero_grad()
        value_loss.backward()  # frees value graph
        optimizer_value.step()

        # --- KL Divergence ---
        kl_div = compute_kl_divergence(dist_old_all, dist_new)
        history["kl_divergences"].append(kl_div.item())

        # Print some logs
        if (epoch + 1) % 10 == 0:
            last_n_rewards = history["episode_rewards"][-10:]
            if len(last_n_rewards) > 0:
                avg_reward = sum(last_n_rewards) / len(last_n_rewards)
            else:
                avg_reward = 0.0
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"AvgReward (last 10): {avg_reward:.2f} | "
                f"KL: {kl_div:.4f} | "
                f"PolicyLoss: {policy_loss.item():.4f} | "
                f"ValueLoss: {value_loss.item():.4f}"
            )

    return history


def supervised_training(model: nn.Module,
                        dataloader,
                        optimizer,
                        criterion,
                        epochs: int = 5):
    """
    Generic supervised training loop for classification or regression tasks.
    Could be for next-token prediction if this is a language model.

    :param model: nn.Module.
    :param dataloader: Prepared DataLoader with labeled data.
    :param optimizer: Torch optimizer.
    :param criterion: Loss function (e.g. CrossEntropy, MSE).
    :param epochs: Number of training epochs.
    :return: Training history dict (e.g., loss per epoch).
    """
    model.train()
    history = {"loss": []}
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # example: (x, y)
            x, y = batch
            x = x.float()
            y = y.long()  # if classification

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        history["loss"].append(avg_loss)
        print(f"[Supervised] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return history


def train_model(model, train_data, optimizer, criterion, epochs=5):
    """
    Wrap `supervised_training` to keep an interface for 'train_model' that's more general.
    Possibly do some pre-processing, or combined logic if needed.
    """
    dataloader = prepare_dataloaders(train_data)
    history = supervised_training(model, dataloader, optimizer, criterion, epochs)
    return history


def evaluate_model(model, eval_data, criterion):
    """
    Evaluate the model performance on some validation or test set.

    :param model: nn.Module.
    :param eval_data: Validation/test dataset or dataloader.
    :param criterion: Loss function for evaluation metric.
    :return: Performance metrics (e.g., accuracy or loss).
    """
    dataloader = prepare_dataloaders(eval_data)
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.float()
            y = y.long()
            logits = model(x)
            loss = criterion(logits, y)
            eval_loss += loss.item()

            # For classification accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = eval_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Eval - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}")
    return {"loss": avg_loss, "accuracy": accuracy}


def supervised_training_plot(history):
    """
    Plot or visualize supervised training results, e.g. training loss curves.
    """
    plt.figure()
    plt.title("Supervised Training Loss")
    plt.plot(history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def grpo_rl_finetuning_generic_plot(history):
    """
    Plot relevant RL curves: reward vs epochs, KL divergence, etc.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    axs[0].plot(history["episode_rewards"], label="Episode Reward")
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Rollout Steps (aggregated by epoch?)")
    axs[0].set_ylabel("Reward")

    axs[1].plot(history["kl_divergences"], label="KL Divergence", color="orange")
    axs[1].set_title("KL Divergence")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("KL")

    plt.tight_layout()
    plt.show()
