import gym
from torch import nn, optim

from grpo_rl_train import grpo_rl_finetuning_generic, grpo_rl_finetuning_generic_plot, SimplePolicy


def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_model = SimplePolicy(state_dim, action_dim)
    value_model = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    optimizer_policy = optim.Adam(policy_model.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_model.parameters(), lr=1e-3)

    history = grpo_rl_finetuning_generic(env,
                                         policy_model,
                                         value_model,
                                         optimizer_policy,
                                         optimizer_value,
                                         epochs=2000,
                                         rollout_steps=2048)
    grpo_rl_finetuning_generic_plot(history)


if __name__ == "__main__":
    main()