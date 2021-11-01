"""PPO in a learned world model. In development.
"""

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

# torch.random.manual_seed(0)
# np.random.seed(0)
plt.style.use(["science", "notebook", "grid", "no-latex"])  # pip install SciencePlots


class Memory:
    def __init__(self, batch_size=32):
        self.states = []
        self.logits = []
        self.values = []
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
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones),
            torch.tensor(self.advantages, dtype=torch.float32),
            batches,
        )

    def store_memory(self, state, action, logits, vals, reward, done, advantage):
        self.states.extend(state)
        self.actions.extend(action)
        self.logits.extend(logits)
        self.values.extend(vals)
        self.rewards.extend(reward)
        self.dones.extend(done)
        self.advantages.extend(advantage)

    def clear_memory(self):
        self.states = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("tanh"))
#         # m.bias.data.fill_(0.01)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0.0)


class Agent(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Agent, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic.apply(init_weights)
        self.actor.apply(init_weights)
        print(self)

    def forward(self, x):
        policy_logits = self.actor(x)
        dist = Categorical(logits=policy_logits)
        value = self.critic(x)

        return dist, value, policy_logits


class Model(nn.Module):
    def __init__(self, obs_space):
        super(Model, self).__init__()

        # predict next state and reward
        self.predict = nn.Sequential(
            nn.Linear(obs_space + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, obs_space + 1),
        )
        # self.predict.apply(init_weights)
        print(self)

    def forward(self, x):
        return self.predict(x)


class PPO:
    def __init__(self, env, agent, model, hyp, training=True):
        self.env = env
        self.REWARD_THRESHOLD = 150
        self.agent = agent
        self.model = model
        self.training = training
        self.EPOCHS = hyp["epochs"]
        self.global_step = 0
        self.env_step = 0
        self.PRINT_FREQUENCY = 1
        self.RENDER = False
        self.TEST_FREQ = hyp["test_freq"]
        self.RESET_STEP = hyp["reset_step"]

        # model
        # self.MODEL_INITIAL_N_EPOCHS = hyp["model_initial_n_epochs"]
        # self.INITIAL_RANDOM_TRAJ = hyp["initial_random_traj"]
        self.MODEL_BATCH_SIZE = hyp["model_batch_size"]
        self.MODEL_N_EPOCHS = hyp["model_n_epochs"]
        self.RESET_STEP = hyp["reset_step"]
        self.model_loss_fn = nn.MSELoss()
        self.MODEL_INITIAL_LR = hyp["model_initial_lr"]
        self.model_opt = optim.Adam(model.parameters(), lr=self.MODEL_INITIAL_LR)
        self.model_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.model_opt,
            "min",
            factor=hyp["model_decay_gamma"],
            patience=10,
            min_lr=1e-5,
            verbose=True,
        )
        # agent
        self.GAMMA = hyp["gamma"]
        self.GAE_LAMBDA = hyp["gae_lambda"]
        self.EPS = hyp["eps"]

        self.TIMESTEPS = hyp["timesteps"]
        self.AGENT_BATCH_SIZE = hyp["agent_batch_size"]
        self.AGENT_N_EPOCHS = hyp["agent_n_epochs"]

        self.C1 = hyp["c1"]  # 0.5  #critic loss coefficient
        self.C2 = hyp["c2"]  # entropy coefficient

        self.agent_opt = torch.optim.Adam(
            [
                {"params": self.agent.actor.parameters(), "lr": hyp["actor_lr"]},
                {"params": self.agent.critic.parameters(), "lr": hyp["critic_lr"]},
            ]
        )
        self.agent_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.agent_opt,
            "min",
            factor=hyp["agent_decay_gamma"],
            patience=10,
            min_lr=1e-5,
            verbose=True,
        )

    def get_random_trajectories(self, n):
        """  Generate n random trajectories """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for _ in range(n):
            state = self.env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                state = next_state
                self.env_step += 1

        return states, actions, rewards, next_states, dones

    def select_action(self, state, training):
        dist, value, policy_logits = self.agent(state)
        if training:
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

        return advantages, targets

    def learn_model(self, epochs, states, actions, rewards, next_states):

        states_actions = torch.cat(
            (
                torch.tensor(states, dtype=torch.float32),
                torch.reshape(torch.tensor(actions, dtype=torch.float32), (-1, 1)),
            ),
            dim=1,
        )
        next_states_rewards = torch.cat(
            (
                torch.tensor(next_states, dtype=torch.float32),
                torch.reshape(torch.tensor(rewards, dtype=torch.float32)/100, (-1, 1)),
            ),
            dim=1,
        )
        data = torch.utils.data.TensorDataset(states_actions, next_states_rewards)

        dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
        step = 0
        for t in range(epochs):
            total_loss = 0
            for batch, (X, y) in enumerate(dataloader):

                next_state_reward_predicts = self.model(X)
                loss = self.model_loss_fn(next_state_reward_predicts, y)

                self.model_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                self.model_opt.step()

                loss = loss.item()
                total_loss += loss

                self.global_step += 1

            writer.add_scalar("training/model_loss", total_loss, self.global_step)
            writer.add_scalar(
                "training/model_lr",
                self.model_opt.param_groups[0]["lr"],
                self.global_step,
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "training/model_loss": total_loss,
                        "training/model_lr": self.model_opt.param_groups[0]["lr"],
                        "time/global_step": self.global_step,
                        "time/env_step": self.env_step,
                    }
                )
        self.model_scheduler.step(total_loss / len(dataloader))
        return total_loss

    def learn(self, targets):

        for epoch in range(self.AGENT_N_EPOCHS):
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
            ) = self.mem.generate_batches()

            action_idx = torch.stack((torch.arange(0, len(rewards)), actions), 1)

            old_probs = nn.functional.softmax(logits_old, dim=1)
            old_probs = old_probs[list(action_idx.T)]
            num_batches = len(batches)

            for batch in batches:
                dist, critic_value, logits = self.agent(states[batch])
                probs = torch.nn.functional.softmax(logits, dim=1)

                action_idx_batch = torch.stack(
                    (torch.arange(0, len(probs)), actions[batch]), 1
                )

                probs = probs[list(action_idx_batch.T)]

                critic_value = torch.squeeze(critic_value)

                entropy = torch.mean(dist.entropy())

                # distribution ratio
                r_theta = torch.exp(probs - torch.squeeze(old_probs[batch]))

                advs = advantages[batch]
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                # policy clipping
                policy_obj = r_theta * advs
                clipped_r_theta = torch.clip(r_theta, 1 - self.EPS, 1 + self.EPS) * advs
                # compute losses
                actor_loss = -torch.mean(torch.minimum(policy_obj, clipped_r_theta))

                # clip value function
                critic_loss_unclipped = torch.square(targets[batch] - critic_value)
                v_loss_clipped = targets[batch] + torch.clamp(
                    critic_value - targets[batch], -0.5, 0.5
                )
                critic_loss_clipped = torch.square(v_loss_clipped - targets[batch])
                v_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                critic_loss = torch.mean(v_loss_max)

                # critic_loss = torch.mean(torch.square(targets[batch] - critic_value))

                loss = actor_loss + self.C1 * critic_loss + self.C2 * entropy

                total_loss += loss
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy

                self.agent_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.agent.parameters(), 0.5)
                self.agent_opt.step()

                self.global_step += 1

            total_loss.detach().numpy()
            total_entropy_loss.detach().numpy()
            total_actor_loss.detach().numpy()
            total_critic_loss.detach().numpy()

            self.loss_hist.append(total_loss)
            self.actor_loss_hist.append(total_actor_loss)
            self.critic_loss_hist.append(total_critic_loss)
            self.entropy_hist.append(total_entropy_loss)
            writer.add_scalar("training/loss", total_loss, self.global_step)
            writer.add_scalar("training/actor_loss", total_actor_loss, self.global_step)
            writer.add_scalar(
                "training/critic_loss", total_critic_loss, self.global_step
            )
            writer.add_scalar(
                "training/entropy_loss", total_entropy_loss, self.global_step
            )
            # writer.add_scalar(
            #     "training/learning_rate",
            #     self.agent_scheduler.get_last_lr()[0],
            #     self.global_step,
            # )
            if self.use_wandb:
                wandb.log(
                    {
                        "training/total_loss": total_loss,
                        "training/actor_loss": total_actor_loss,
                        "training/entropy": total_entropy_loss,
                        "training/critic_loss": total_critic_loss,
                        # "training/learning_rate": self.agent_scheduler.get_last_lr()[0],
                        "time/epoch": self.epoch,
                        "time/env_step": self.env_step,
                        "time/global_step": self.global_step,
                    }
                )
        self.agent_scheduler.step(total_loss / len(batches))

    def train(self):
        self.episode_reward_hist = []
        self.true_episode_reward_hist = []
        self.loss_hist = []
        self.entropy_hist = []
        self.actor_loss_hist = []
        self.critic_loss_hist = []

        done = False
        episode_reward = 0
        true_episode_reward = 0
        episode = 1

        for epoch in range(self.EPOCHS):
            state = env.reset()
            rewards = []
            true_rewards = []
            actions = []
            values = []
            states = []
            next_states = []
            true_next_states = []
            dones = []
            policy_logits = []
            self.epoch = epoch
            self.training = True
            self.mem = Memory(self.AGENT_BATCH_SIZE)


            for t in range(self.TIMESTEPS):

                if self.RENDER:
                    self.env.render()

                # select action and step in learned model
                action, value, pls = self.select_action(
                    torch.tensor(state, dtype=torch.float32), self.training
                )

                next_state_reward_predicts = self.model(
                    torch.tensor(np.hstack((state, action)), dtype=torch.float32)
                )
                # also step in real model
                true_next_state, true_reward, true_done, _ = self.env.step(action)

                next_state = (
                    next_state_reward_predicts[: env.observation_space.shape[0]]
                    .detach()
                    .numpy()
                )
                reward = next_state_reward_predicts[-1].detach().item()*100

                episode_reward += reward
                true_episode_reward += true_reward

                actions.append(action)
                values.append(value.detach().item())
                rewards.append(reward)
                true_rewards.append(true_reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(true_done)
                policy_logits.append(pls.detach().numpy())
                true_next_states.append(true_next_state)

                if true_done:
                    state = true_next_state = env.reset()
                    self.episode_reward_hist.append(episode_reward)
                    self.true_episode_reward_hist.append(true_episode_reward)
                    writer.add_scalar("training/reward", episode_reward)
                    writer.add_scalar("training/true_reward", true_episode_reward)
                    if self.use_wandb:
                        wandb.log(
                            {
                                "training/true_reward": true_episode_reward,
                                "training/reward": episode_reward,
                                "time/global_step": self.global_step,
                                "time/env_step": self.env_step,
                                "time/epoch": self.epoch,
                            }
                        )
                    episode_reward = 0
                    true_episode_reward = 0
                    episode += 1

                else:
                    state = next_state

                if t % self.RESET_STEP == 0:
                    state = true_next_state
                    next_states[-1] = true_next_state
                    rewards[-1] = true_reward

                self.global_step += 1

            if epoch % 3 == 0:
                pltstates = np.array(states)
                plttruestates = np.array(true_next_states)
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(pltstates[:, 0], pltstates[:, 1], '.', label='states')
                plt.plot(plttruestates[:, 0], plttruestates[:, 1], 'x', label='true_states')
                plt.subplot(2,1,2)  
                plt.hist([tr-r for r,tr in zip(true_rewards, rewards)], bins=100, log=True)
                plt.ylabel("Count")
                plt.xlabel("Reward error")
                plt.legend()
                plt.savefig(f'tmp/states_rewards_{epoch}.png')
                plt.close()
                plt.figure()
                plt.plot(pltstates[:, 0], pltstates[:, 1], '.', label='predicted location')
                plt.plot(plttruestates[:, 0], plttruestates[:, 1], 'x', label='true location')
                plt.legend()
                plt.savefig(f'tmp/states_{epoch}.png')
                plt.close()

            _, next_values, _ = self.agent(
                torch.tensor(next_states, dtype=torch.float32)
            )

            _, true_next_values, _ = self.agent(
                torch.tensor(true_next_states, dtype=torch.float32)
            )

            advantages, targets = self.compute_advantages(
                values, next_values.detach().numpy(), rewards, dones
            )

            self.mem.store_memory(
                states, actions, policy_logits, values, rewards, dones, advantages
            )
            writer.add_histogram("training/values", np.array(values), self.global_step)
            writer.add_histogram(
                "training/rewards", np.array(rewards), self.global_step
            )
            writer.add_histogram(
                "training/true_rewards", np.array(true_rewards), self.global_step
            )
            writer.add_histogram("training/next_values", next_values, self.global_step)
            writer.add_histogram("training/advantages", advantages, self.global_step)
            writer.add_histogram("training/targets", targets, self.global_step)
            writer.add_histogram(
                "training/next_states", np.array(next_states), self.global_step
            )
            writer.add_histogram(
                "training/true_pred_next_state_diff",
                np.array(
                    [
                        np.linalg.norm(tns - ns)
                        for tns, ns in zip(true_next_states, next_states)
                    ]
                ),
                self.global_step,
            )
            writer.add_histogram(
                "training/true_next_states",
                np.array(true_next_states),
                self.global_step,
            )
            self.learn(torch.tensor(targets))

            self.mem.clear_memory()

            # states, actions, rewards, next_states, dones = self.get_random_trajectories(20)
            # model_loss = self.learn_model(30, states, actions, rewards, next_states)

            _, states, actions, rewards, next_states = self.test(30)
            model_loss = self.learn_model(
                self.MODEL_N_EPOCHS, states, actions, rewards, next_states
            )

            if epoch % self.PRINT_FREQUENCY == 0:
                mean_reward = np.mean(self.episode_reward_hist[-50:])
                mean_true_reward = np.mean(self.true_episode_reward_hist[-50:])
                print(
                    f"Epoch: {epoch}\tAgent Loss: {self.loss_hist[-1]:.2e}\tModel Loss {model_loss:.2e} Reward: {self.episode_reward_hist[-1]:.2e}\tTrue Reward: {self.true_episode_reward_hist[-1]:.2e}\tMean reward: {mean_reward:.2e}\tMean true reward: {mean_true_reward:.2e}"
                )

            if (epoch + 1) % self.TEST_FREQ == 0:
                self.training = False
                test_reward, _, _, _, _ = self.test(30)

                if test_reward > self.REWARD_THRESHOLD:
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
        path = f"./saved_models/saved_mbrl_ppo_model_torch_{dt_str}"
        os.mkdir(path)
        torch.save(self.agent.state_dict(), f"{path}/agent")
        torch.save(self.model.state_dict(), f"{path}/agent")
        self.save_ppo_toml(path)
        print("Model saved")
        return path

    def load(self, agent_path, model_path):
        self.agent = torch.load(agent_path)
        print(f"Model {agent_path} loaded")
        self.model = torch.load(model_path)
        print(f"Model {model_path} loaded")

    def test(self, episodes, render=False):

        # print("Testing agent")

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        frames = []
        test_episode_reward_hist = []
        self.mem = Memory(self.MODEL_BATCH_SIZE)

        if render:

            fig = plt.figure()

        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                action, value, logits = self.select_action(
                    torch.tensor(state, dtype=torch.float32), True
                )
                next_state, reward, done, _ = self.env.step(action)
                # reward /= 100
                # record data
                actions.append(action)
                # NOTE trying scaling rewards
                rewards.append(reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(done)

                if render:
                    img = plt.imshow(self.env.render("rgb_array"))
                    plt.grid()
                    frames.append([img])

                state = next_state
                episode_reward += reward
                step += 1

            # print(f"Episode {episode} lasted {step} steps. Reward: {episode_reward}")
            test_episode_reward_hist.append(episode_reward)
            writer.add_scalar("test/test_reward", episode_reward, self.global_step)
            if self.use_wandb:
                wandb.log(
                    {
                        "test/test_reward": episode_reward,
                        "time/global_step": self.global_step,
                        "time/env_step": self.env_step,
                    }
                )
        if self.use_wandb:
            wandb.log(
                {
                    "test/mean_test_reward": np.mean(test_episode_reward_hist),
                    "time/global_step": self.global_step,
                    "time/env_step": self.env_step,
                    "time/epoch": self.epoch,
                }
            )

        if render:
            an = animation.ArtistAnimation(
                fig, frames, interval=100, repeat_delay=1000, blit=True
            )
            writergif = animation.PillowWriter(fps=30)
            an.save("animation.gif", writer=writergif)
        return np.mean(test_episode_reward_hist), states, actions, rewards, next_states


if __name__ == "__main__":

    # gpu not implemented yet
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"using device {device}")
    writer = SummaryWriter(f"logs/{datetime.now().strftime('%d%b%y_%H.%M')}")

    env_name = "LunarLander-v2"
    # env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    model = Model(env.observation_space.shape[0])

    use_wandb = False

    hyperparameter_defaults = dict(
        ##### general
        env=env_name,
        epochs=500,
        test_freq=20,
        ##### model
        model_batch_size=128,
        model_n_epochs=30,
        model_initial_lr=0.001,
        model_decay_gamma=0.95,
        reset_step=5,
        ##### agent
        gamma=0.99,
        gae_lambda=0.95,
        eps=0.2,
        agent_batch_size=128,
        timesteps=2000,
        agent_n_epochs=40,
        c1=0.5,
        c2=0.001,
        actor_lr=3e-4,
        critic_lr=1e-3,
        agent_decay_gamma=0.95,
    )

    writer.add_hparams(hyperparameter_defaults, {})
    if use_wandb:
        wandb.init(
            config=hyperparameter_defaults,
            project=f"mbrl",
            group="ppo_in_learned_model",
            # name="",
            notes=f"testing initial script {env_name} {datetime.now().strftime('%d%b%y %H:%M')}",
            tags=["testing"],
            magic=True,
            save_code=True,
        )
        config = wandb.config
        # wandb.watch(agent, log_freq=100)

    ppo = PPO(env, agent, model, hyperparameter_defaults)
    ppo.use_wandb = use_wandb

    print("initial model learning")
    # states, actions, rewards, next_states, dones = ppo.get_random_trajectories(200)
    # ppo.learn_model(20, states, actions, rewards, next_states)

    print("training agent")
    ppo.train()
    wandb.finish()

    # ppo.plot_training()
    # path = ppo.save()
    # ppo.load("./saved_ppo_model_2021-08-28T14-02-10/agent")

    # ppo.test(10, render=False)
    # ppo.test(1, render=True)
