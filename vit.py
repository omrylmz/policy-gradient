import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# =======================================================
# 1. Simple Vision Transformer Backbone for CIFAR-10
# =======================================================
class SimpleViTBackbone(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super(SimpleViTBackbone, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.patch_size = patch_size
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        # Linear projection of flattened patches
        self.patch_embedding = nn.Linear(patch_size * patch_size * in_channels, dim)
        # Learnable class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        patch_size = self.patch_size
        # Break image into patches
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # x: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)
        x = x.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, patch_size, patch_size)
        x = x.contiguous().view(B, self.num_patches, -1)  # flatten each patch
        x = self.patch_embedding(x)  # (B, num_patches, dim)

        # Prepend the class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, dim)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)  # (B, num_patches+1, dim)
        return x[:, 0]  # return the class token representation (B, dim)


# =======================================================
# 2. Actor and Critic Heads Using the ViT Backbone
# =======================================================
class ViTActor(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super(ViTActor, self).__init__()
        self.backbone = backbone
        self.actor_head = nn.Linear(backbone.dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.actor_head(features)
        return logits


class ViTCritic(nn.Module):
    def __init__(self, backbone):
        super(ViTCritic, self).__init__()
        self.backbone = backbone
        self.critic_head = nn.Linear(backbone.dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        value = self.critic_head(features)
        return value


# =======================================================
# 3. Memory for Storing Trajectories (RL)
# =======================================================
class Memory:
    def __init__(self):
        self.states = []  # list of image tensors (or numpy arrays)
        self.actions = []  # list of scalar actions
        self.logprobs = []  # list of log probabilities
        self.rewards = []  # list of rewards
        self.is_terminals = []  # list of terminal flags

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


# =======================================================
# 4. PPOAgentViT and GRPOAgentViT
# =======================================================
class PPOAgentViT:
    def __init__(self, num_classes=10, lr=3e-4, gamma=0.99, eps_clip=0.2, beta=0.01, image_size=32):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.beta = beta

        backbone = SimpleViTBackbone(image_size=image_size, patch_size=4, in_channels=3,
                                     dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1)
        self.actor = ViTActor(backbone, num_classes)
        self.critic = ViTCritic(backbone)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        # Create old policy
        self.actor_old = ViTActor(backbone, num_classes)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def select_action(self, state):
        # state is an image; ensure it is a float tensor with shape (1, C, H, W)
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
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
        # Use np.stack to combine list of tensors for efficiency
        states = torch.tensor(np.stack(memory.states), dtype=torch.float32)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.long)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32)

        # Compute discounted rewards
        discounted_rewards = []
        reward_sum = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                reward_sum = 0
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.insert(0, reward_sum)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
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


class GRPOAgentViT(PPOAgentViT):
    """
    GRPOAgentViT extends PPOAgentViT by adding a gradient regularization term.
    """

    def __init__(self, num_classes=10, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 beta=0.01, lambda_grad=1e-3, image_size=32):
        super(GRPOAgentViT, self).__init__(num_classes, lr, gamma, eps_clip, beta, image_size)
        self.lambda_grad = lambda_grad

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
        grad_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
        states.requires_grad_(False)
        return grad_penalty

    def update(self, memory, K_epochs=4):
        states = torch.tensor(np.stack(memory.states), dtype=torch.float32)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.long)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32)

        discounted_rewards = []
        reward_sum = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                reward_sum = 0
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.insert(0, reward_sum)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
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


# =======================================================
# 5. Dummy Vision Environment for RL Fine-Tuning
# =======================================================
class DummyVisionEnv:
    """
    A dummy environment that iterates over a dataset of images.
    On reset, it returns a random image and its label.
    On step, it returns the next image and gives a reward of 1 if the
    predicted action (label) matches the true label, else 0.
    An episode lasts for a fixed number of steps.
    """

    def __init__(self, dataset, max_steps=10):
        self.dataset = dataset
        self.max_steps = max_steps
        self.current_step = 0
        self.index = 0

    def reset(self):
        self.current_step = 0
        self.index = np.random.randint(len(self.dataset))
        image, label = self.dataset[self.index]
        return image, label

    def step(self, action):
        image, label = self.dataset[self.index]
        reward = 1.0 if action == label else 0.0
        self.current_step += 1
        done = self.current_step >= self.max_steps
        self.index = (self.index + 1) % len(self.dataset)
        next_image, next_label = self.dataset[self.index]
        return next_image, reward, done, {"label": next_label}


# =======================================================
# 6. BaseFinetuner: Modular Functions for Supervised & RL Finetuning
# =======================================================
class BaseFinetuner:
    def __init__(self, agent, supervised_dataset, batch_size=32, validation_split=0.2):
        """
        agent: instance of PPOAgentViT or GRPOAgentViT
        supervised_dataset: a torchvision dataset (e.g. CIFAR10) for supervised training
        """
        self.agent = agent
        self.supervised_dataset = supervised_dataset
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
        """Prepare training and validation DataLoaders from the supervised dataset."""
        dataset = self.supervised_dataset
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

    def extract_logits(self, x):
        x = x.to(self.device)
        logits = self.agent.actor(x)
        return logits

    def get_probability_distribution(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return Categorical(probs)

    def default_reward_function(self, predictions, targets):
        loss = nn.CrossEntropyLoss()(predictions, targets.to(predictions.device))
        return -loss

    def sample_group_info(self, data, group_size=10):
        indices = torch.randperm(len(data))[:group_size]
        return indices

    def compute_advantages(self, rewards, values):
        advantages = rewards - values.detach()
        return advantages

    def compute_surrogate_loss(self, ratios, advantages):
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.agent.eps_clip, 1 + self.agent.eps_clip) * advantages
        return -torch.min(surr1, surr2).mean()

    def compute_kl_divergence(self, states):
        return self.agent.compute_kl(states)

    def rl_finetuning_generic(self, env, num_episodes=100, update_timestep=20):
        """
        Perform RL fine-tuning using the agent's update method in the given environment.
        Here, env is expected to be an instance of DummyVisionEnv.
        """
        memory = Memory()
        timestep = 0
        self.rl_reward_history = []

        for episode in range(1, num_episodes + 1):
            state, label = env.reset()
            # state is an image; label is the true label (for reward computation)
            ep_reward = 0
            done = False

            while not done:
                timestep += 1
                action, logprob, _ = self.agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                memory.states.append(state.cpu().numpy())  # store as numpy array
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

    def supervised_training(self, num_epochs=10, learning_rate=1e-3):
        self.prepare_dataloaders()
        optimizer = optim.Adam(self.agent.actor.parameters(), lr=learning_rate)
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

    def train_model(self, supervised_epochs=10, rl_episodes=100, rl_env=None):
        print("Starting supervised training...")
        self.supervised_training(num_epochs=supervised_epochs)
        if rl_env is not None:
            print("Starting RL fine-tuning...")
            self.rl_finetuning_generic(rl_env, num_episodes=rl_episodes)

    def evaluate_model(self):
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
        plt.figure(figsize=(8, 4))
        plt.plot(self.rl_reward_history, marker='o', label='RL Finetuning Reward')
        if len(self.rl_reward_history) >= 10:
            weighted_avg = np.convolve(self.rl_reward_history, np.ones(10) / 10, mode='valid')
            plt.plot(np.arange(9, len(self.rl_reward_history)), weighted_avg, color='red',
                     label='Weighted Avg (10 episodes)')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("RL Finetuning Reward")
        plt.legend()
        plt.show()


# =======================================================
# 7. Example Usage: Training with PPOAgentViT and GRPOAgentViT on CIFAR-10
# =======================================================
if __name__ == '__main__':
    # Prepare CIFAR-10 training dataset for supervised training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a dummy vision environment from the test set for RL fine-tuning
    dummy_env = DummyVisionEnv(cifar_test, max_steps=10)

    # --- Training with PPOAgentViT ---
    print("\n--- Training with PPOAgentViT ---")
    ppo_agent_vit = PPOAgentViT(num_classes=10, image_size=32)
    ppo_finetuner = BaseFinetuner(ppo_agent_vit, cifar_train, batch_size=64, validation_split=0.2)
    ppo_finetuner.train_model(supervised_epochs=10, rl_episodes=50, rl_env=dummy_env)
    ppo_finetuner.evaluate_model()
    ppo_finetuner.supervised_training_plot()
    ppo_finetuner.rl_finetuning_generic_plot()

    # --- Training with GRPOAgentViT ---
    print("\n--- Training with GRPOAgentViT ---")
    grpo_agent_vit = GRPOAgentViT(num_classes=10, lambda_grad=1e-3, image_size=32)
    grpo_finetuner = BaseFinetuner(grpo_agent_vit, cifar_train, batch_size=64, validation_split=0.2)
    grpo_finetuner.train_model(supervised_epochs=10, rl_episodes=50, rl_env=dummy_env)
    grpo_finetuner.evaluate_model()
    grpo_finetuner.supervised_training_plot()
    grpo_finetuner.rl_finetuning_generic_plot()
