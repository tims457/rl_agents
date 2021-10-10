import os
import random
import time
import torch
import gym
import copy
import toml
import json
import wandb
from datetime import datetime
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
            torch.tensor(self.states, dtype=torch.float32, device=device),
            torch.tensor(self.actions, device=device),
            torch.tensor(self.logits, dtype=torch.float32, device=device),
            torch.tensor(self.vals, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones),
            torch.tensor(self.advantages, dtype=torch.float32, device=device),
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
        print(self)

    def forward(self, x):
        policy_logits = self.actor(x)
        dist = Categorical(logits=policy_logits)
        value = self.critic(x)

        return dist, value, policy_logits


class PPO:
    def __init__(self, env, agent, hyp, training=True):
        self.env = env
        self.reward_threshold = 150
        self.agent = agent
        self.training = training

        self.EPOCHS = hyp["epochs"]
        self.GAMMA = hyp["gamma"]
        self.GAE_LAMBDA = hyp["gae_lambda"]
        self.EPS = hyp["eps"]

        self.TIMESTEPS = hyp["timesteps"]
        self.BATCH_SIZE = hyp["batch_size"]
        self.N_EPOCHS = hyp["n_epochs"]

        self.C1 = hyp["c1"]  # 0.5  #critic loss coefficient
        self.C2 = hyp["c2"]  # entropy coefficient

        self.initial_learning_rate = hyp["lr0"]
        self.opt = optim.Adam(agent.parameters(), lr=self.initial_learning_rate)
        # self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.opt, lr_lambda=lambda x: 0.99)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[100, 200, 400, 9e9], gamma=hyp["lr_decay"]
        )
        self.decay_lr = True
        self.step = 0

        self.print_frequency = 1
        self.render = False
        self.use_wandb = False

    def select_action(self, state):
        dist, value, policy_logits = self.agent(state)
        if self.training:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs)
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

            action_idx = torch.stack(
                (torch.arange(0, len(rewards), device=device), actions), 1
            )

            old_probs = nn.functional.softmax(logits_old, dim=1)
            old_probs = old_probs[list(action_idx.T)]
            old_probs.to(device)
            self.agent.to(device)

            for batch in batches:
                dist, critic_value, logits = self.agent(states[batch])
                probs = torch.nn.functional.softmax(logits, dim=1)

                action_idx_batch = torch.stack(
                    (torch.arange(0, len(probs), device=device), actions[batch]), 1
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

        self.scheduler.step()

        self.loss_hist.append(total_loss.cpu().detach().numpy())
        self.actor_loss_hist.append(total_actor_loss.cpu().detach().numpy())
        self.critic_loss_hist.append(total_critic_loss.cpu().detach().numpy())
        self.entropy_hist.append(total_entropy_loss.cpu().detach().numpy())
        writer.add_scalar("losses/loss", total_loss, self.step)
        writer.add_scalar("losses/actor_loss", total_actor_loss, self.step)
        writer.add_scalar("losses/critic_loss", total_critic_loss, self.step)
        writer.add_scalar("losses/entropy_loss", total_entropy_loss, self.step)
        writer.add_scalar(
            "opt/learning_rate", self.scheduler.get_last_lr()[0], self.step
        )
        if self.use_wandb:
            wandb.log(
                {
                    "total_loss": total_loss,
                    "actor_loss": total_actor_loss,
                    "entropy": total_entropy_loss,
                    "critic_loss": total_critic_loss,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                }
            )

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
                    if self.use_wandb:
                        wandb.log(
                            {
                                "reward": episode_reward,
                                "mean_reward": np.mean(self.episode_reward_hist[-20:]),
                            }
                        )
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
            self.learn(mem, torch.tensor(targets, device=device))

            mem.clear_memory()
            self.agent.to("cpu")

            if epoch % self.print_frequency == 0:
                mean_reward = np.mean(self.episode_reward_hist[-50:])
                print(
                    f"Epoch: {epoch}\tLoss: {self.loss_hist[-1]:.2f}\tReward: {self.episode_reward_hist[-1]:.2f}\tMean reward: {mean_reward:.2f}\tLearning rate: {self.opt.param_groups[0]['lr']:.3e}"
                )

            if epoch % 20 == 0 and epoch > 50:
                _, test_reward = self.test(30)

                if test_reward > self.reward_threshold:
                    print(f"Solved in {epoch} epochs")
                    break
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

    def save_ppo_toml(self, path):
        with open(f"{path}/ppo.text", "w") as f:
            f.write(str(ppo.__dict__))

    def save(self):
        dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = f"./saved_models/saved_ppo_model_torch_{dt_str}"
        os.mkdir(path)
        torch.save(self.agent.state_dict(), f"{path}/agent")
        self.save_ppo_toml(path)
        print("Model saved")
        return path

    def load(self, PATH):
        self.agent = torch.load(PATH)
        print(f"Model {PATH} loaded")

    def test(self, episodes, render=False):

        print("Testing agent")

        self.training = False
        mem = Memory()
        frames = []
        test_episode_reward_hist = []

        if render:

            fig = plt.figure()

        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                action, value, logits = self.select_action(
                    torch.tensor(state, dtype=torch.float32)
                )

                new_state, reward, done, info = self.env.step(action)
                mem.store_memory(
                    [state], [action], [logits], [value], [reward], [done], [None]
                )

                if render:
                    img = plt.imshow(self.env.render("rgb_array"))
                    plt.grid()
                    frames.append([img])

                state = new_state
                episode_reward += reward
                step += 1

            print(f"Episode {episode} lasted {step} steps. Reward: {episode_reward}")
            test_episode_reward_hist.append(episode_reward)
            writer.add_scalar("rewards/reward", episode_reward, self.step)
            if self.use_wandb:
                wandb.log({"test_reward": episode_reward})
        if self.use_wandb:
            wandb.log({"mean_test_reward": np.mean(test_episode_reward_hist)})
        if render:
            an = animation.ArtistAnimation(
                fig, frames, interval=100, repeat_delay=1000, blit=True
            )
            writergif = animation.PillowWriter(fps=30)
            an.save("animation.gif", writer=writergif)
        return mem, np.mean(test_episode_reward_hist)


if __name__ == "__main__":

    # NOTE cuda currently slower than cpu, may have the wrong setup or data movement is slower
    # than gpu speedup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"using device {device}")
    writer = SummaryWriter(f"logs/{datetime.now().strftime('%d%b%y_%H.%M')}")

    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    # env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    use_wandb = True

    hyperparameter_defaults = dict(
        gamma=0.99,
        gae_lambda=0.95,
        eps=0.1,
        batch_size=32,
        timesteps=2000,
        n_epochs=7,
        c1=1,
        c2=0.001,
        lr0=2.5e-4,
        decay_lr=True,
        lr_decay=0.9,
        epochs=500,
    )
    if use_wandb:
        wandb.init(config=hyperparameter_defaults, project="torch_ppo")
        config = wandb.config
        wandb.watch(agent, log_freq=100)

    # writer.add_graph(agent, None, verbose=True)
    ppo = PPO(env, agent, hyperparameter_defaults)
    ppo.use_wandb = use_wandb

    ppo.train()
    wandb.finish()

    # ppo.plot_training()
    # path = ppo.save()
    # ppo.load("./saved_ppo_model_2021-08-28T14-02-10/agent")

    # ppo.test(10, render=False)
    # ppo.test(1, render=True)
