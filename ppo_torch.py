import os
import random
import time
import torch
import gym
import copy
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation, rc
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


torch.random.manual_seed(0)
np.random.seed(0)
plt.style.use(["science", "notebook", "grid", "no-latex"])  # pip install SciencePlots


class Memory:
    def __init__(self, batch_size=32):
        self.states = []
        self.logits = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.advantages = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            torch.tensor(self.states, dtype=torch.float32),
            torch.tensor(self.actions),
            torch.tensor(self.logits, dtype=torch.float32),
            torch.tensor(self.vals, dtype=torch.float32),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones),
            torch.tensor(self.advantages, dtype=torch.float32),
            batches,
        )

    def store_memory(self, state, action, logits, vals, reward, done, advantage):
        self.states.extend(state)
        self.actions.extend(action)
        self.logits.extend(logits)
        self.vals.extend(vals)
        self.rewards.extend(reward)
        self.dones.extend(done)
        self.advantages.extend(advantage)

    def clear_memory(self):
        self.states = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.advantages = []


class Agent(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Agent, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        policy_logits = self.actor(x)
        dist = Categorical(logits=policy_logits)
        value = self.critic(x)

        return dist, value, policy_logits


class PPO:
    def __init__(self, env, agent, training=True):
        self.env = env
        self.reward_threshold = 150
        self.agent = agent
        self.training = training

        self.EPOCHS = 200
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.EPS = 0.1

        self.TIMESTEPS = 1000
        self.BATCH_SIZE = 32
        self.N_EPOCHS = 7

        self.C1 = 0.5  # 0.5  #critic loss coefficient
        self.C2 = 0.001  # entropy coefficient

        self.initial_learning_rate = 2.5e-4
        self.opt = optim.Adam(
            agent.parameters(), lr=self.initial_learning_rate
        )  # TODO lr decay
        self.decay_lr = False
        self.step = 0

        self.print_frequency = 1
        self.render = False

    def select_action(self, state):
        dist, value, policy_logits = self.agent(state)
        if self.training:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dims=1)
        return action.item(), value, policy_logits

    def compute_advantages(self, values, next_values, rewards, dones):

        deltas = np.zeros((len(rewards)))
        for t, (r, v, nv, d) in enumerate(zip(rewards, values, next_values, dones)):
            deltas[t] = r + self.GAMMA * (1 - d) * nv - v

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            advantages[t] = (
                advantages[t]
                + (1 - dones[t]) * self.GAMMA * self.GAE_LAMBDA * advantages[t + 1]
            )

        targets = advantages + values

        # normalize advantages
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + 1e-8

        return advantages, targets

    def learn(self, mem, targets):

        for epoch in range(self.N_EPOCHS):
            total_loss = 0
            total_entropy_loss = 0
            total_actor_loss = 0
            total_critic_loss = 0

            (
                states,
                actions,
                logits_old,
                values,
                rewards,
                dones,
                advantages,
                batches,
            ) = mem.generate_batches()

            action_idx = torch.stack((torch.arange(0, len(rewards)), actions), 1)

            old_probs = nn.functional.softmax(logits_old)
            old_probs = old_probs[list(action_idx.T)]

            for batch in batches:
                dist, critic_value, logits = self.agent(states[batch])
                probs = torch.nn.functional.softmax(logits)

                action_idx_batch = torch.stack(
                    (torch.arange(0, len(probs)), actions[batch]), 1
                )

                probs = probs[list(action_idx_batch.T)]

                critic_value = torch.squeeze(critic_value)

                entropy = torch.mean(dist.entropy())

                # distribution ratio
                r_theta = torch.exp(probs - torch.squeeze(old_probs[batch]))

                # policy clipping
                policy_obj = r_theta * advantages[batch]
                clipped_r_theta = (
                    torch.clip(r_theta, 1 - self.EPS, 1 + self.EPS) * advantages[batch]
                )
                # compute losses
                actor_loss = -torch.mean(torch.minimum(policy_obj, clipped_r_theta))

                critic_loss = torch.mean(torch.square(targets[batch] - critic_value))

                loss = actor_loss + self.C1 * critic_loss + self.C2 * entropy

                total_loss += loss
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.step += 1

        self.loss_hist.append(total_loss.detach().numpy())
        self.actor_loss_hist.append(total_actor_loss.detach().numpy())
        self.critic_loss_hist.append(total_critic_loss.detach().numpy())
        self.entropy_hist.append(total_entropy_loss.detach().numpy())
        writer.add_scalar("losses/loss", total_loss, self.step)
        writer.add_scalar("losses/actor_loss", total_actor_loss, self.step)
        writer.add_scalar("losses/critic_loss", total_critic_loss, self.step)
        writer.add_scalar("losses/entropy_loss", total_entropy_loss, self.step)

        if self.decay_lr and self.epoch % 50 == 0 and self.epoch > 0:
            self.opt.learning_rate = self.opt.learning_rate.numpy() * 0.5

    def train(self):

        self.episode_reward_hist = []
        self.loss_hist = []
        self.entropy_hist = []
        self.actor_loss_hist = []
        self.critic_loss_hist = []
        mem = Memory(self.BATCH_SIZE)

        state = env.reset()

        step = 0
        done = False
        episode_reward = 0
        episode = 1

        for epoch in range(self.EPOCHS):
            rewards = []
            actions = []
            values = []
            states = []
            next_states = []
            dones = []
            policy_logits = []
            self.epoch = epoch
            self.training = True

            for t in range(self.TIMESTEPS):

                if self.render:
                    self.env.render()

                action, value, pls = self.select_action(
                    torch.tensor(state, dtype=torch.float32)
                )
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                actions.append(action)
                values.append(value.detach().item())
                rewards.append(reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(done)
                policy_logits.append(pls.detach().numpy())

                if done:
                    state = env.reset()
                    step = 0
                    self.episode_reward_hist.append(episode_reward)
                    writer.add_scalar("rewards/reward", episode_reward, episode)
                    episode_reward = 0
                    episode += 1

                else:
                    step += 1
                    state = next_state

            _, next_values, _ = self.agent(
                torch.tensor(next_states, dtype=torch.float32)
            )

            advantages, targets = self.compute_advantages(
                values, next_values.detach().numpy(), rewards, dones
            )

            mem.store_memory(
                states, actions, policy_logits, values, rewards, dones, advantages
            )
            self.learn(mem, torch.tensor(targets))

            mem.clear_memory()

            if epoch % self.print_frequency == 0:
                mean_reward = np.mean(self.episode_reward_hist[-50:])
                print(
                    f"Epoch: {epoch}\tLoss: {self.loss_hist[-1]:.2f}\tReward: {self.episode_reward_hist[-1]:.2f}\tMean reward: {mean_reward:.2f}\tLearning rate: {self.opt.param_groups[0]['lr']:.3e}"
                )

            if epoch % 20 == 0 and epoch > 50:
                # _, test_reward = self.test(10)

                # if test_reward > self.reward_threshold:
                #     print(f"Solved in {epoch} epochs")
                #     break
                pass

    def plot_training(self):

        window = 10
        plot_alpha = 0.7
        blue = "#4184f3"
        style = ":"

        # loss and reward
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha)
        plt.plot(
            Series(self.loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )

        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.subplot(2, 1, 2)
        plt.plot(
            self.episode_reward_hist,
            blue,
            linestyle=style,
            label="Reward",
            alpha=plot_alpha,
        )
        plt.plot(
            Series(self.episode_reward_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Episode reward")
        plt.xlabel("Training episode")

        plt.savefig("training_loss_reward.png")

        # diagnostics
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        plt.plot(
            self.actor_loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.actor_loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Actor loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 2)
        plt.plot(
            self.critic_loss_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.critic_loss_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Critic loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 3)
        plt.plot(
            self.entropy_hist, blue, linestyle=style, label="Loss", alpha=plot_alpha
        )
        plt.plot(
            Series(self.entropy_hist).rolling(window).mean(),
            blue,
            label=f"Rolling mean, {window}",
        )
        plt.ylabel("Entropy loss")
        plt.xlabel("Epochs")

        plt.savefig("training_loss_diag.png")

        plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    writer = SummaryWriter(f"logs/{dt.datetime.now().strftime('%d%b%y_%H.%M')}")

    env = gym.make("LunarLander-v2")
    # env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    # writer.add_graph(agent, None, verbose=True)
    ppo = PPO(env, agent)

    ppo.train()

    ppo.plot_training()
    # path = ppo.save() #TODO save
    # TODO load

    # ppo.load("./saved_ppo_model_2021-08-28T14-02-10")

    # ppo.test(10, render=False)
    # ppo.test(1, render=True)
