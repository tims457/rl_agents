#!/usr/bin/env python
# coding: utf-8

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import numpy as np
import copy
import toml
from datetime import datetime
from pandas import Series
import json
from envs.cartpole_v0 import ContinuousCartPoleEnv

tf.random.set_seed(0)
np.random.seed(0)
plt.style.use(['science', 'notebook', 'grid',
               'no-latex'])  #pip install SciencePlots


class Memory:

    def __init__(self, batch_size=32):
        self.states = []
        self.probs = []
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
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(
            self.probs), np.array(self.vals), np.array(self.rewards), np.array(
                self.dones), np.array(self.advantages), batches

    def store_memory(self, state, action, probs, vals, reward, done, advantage):
        self.states.extend(state)
        self.actions.extend(action)
        self.probs.extend(probs)
        self.vals.extend(vals)
        self.rewards.extend(reward)
        self.dones.extend(done)
        self.advantages.extend(advantage)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.advantages = []


class Agent(tf.keras.Model):

    def __init__(self):
        super(Agent, self).__init__()

        self.actor = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(
                64,
                activation="tanh",
            ),
            layers.Dense(
                64,
                activation="tanh",
            ),
            layers.Dense(64,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(env.action_space.shape[0] * 2, activation="tanh"),
        ])

        self.critic = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(
                64,
                activation="tanh",
            ),
            layers.Dense(
                64,
                activation="tanh",
            ),
            layers.Dense(64,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(1, activation="linear")
        ])

    def call(self, inputs):

        policy_logits = self.actor(inputs)
        # policy logits are mean and std of distribution
        # dist = tfd.Normal(loc=policy_logits[:, 0], scale=tf.exp(policy_logits[:, 1]),
        # name="policy_dist")
        dist = tfd.MultivariateNormalDiag(
            loc=policy_logits[:, :env.action_space.shape[0]],
            scale_diag=tf.exp(policy_logits[:, env.action_space.shape[0]:]),
            name="policy_dist")

        value = self.critic(inputs)

        return dist, value


class PPO():

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
        self.N_EPOCHS = 10

        self.C1 = 0.5  #critic loss coefficient
        self.C2 = 0.001  #entropy coefficient

        self.initial_learning_rate = 2.5e-4
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=self.initial_learning_rate)
        self.decay_lr = False
        self.step = 0

        # self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=self.initial_learning_rate,
        #     decay_steps=10000,
        #     decay_rate=0.93)

        self.print_frequency = 1
        self.render = False

    def select_action(self, state):
        dist, value = self.agent(tf.expand_dims(state, axis=0))

        if self.training:
            action = tf.squeeze(dist.sample()).numpy()
        else:
            action = tf.squeeze(dist.mean()).numpy()

        return action, tf.squeeze(value), tf.squeeze(dist.log_prob(action))

    def compute_advantages(self, values, next_values, rewards, dones):

        values = tf.cast(values, dtype=tf.float32)

        deltas = np.zeros((len(rewards)))
        for t, (r, v, nv, d) in enumerate(
                zip(rewards, values.numpy(), next_values.numpy(), dones)):
            deltas[t] = r + self.GAMMA * (1 - d) * nv - v

        advantages = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            advantages[t] = advantages[t] + (
                1 - dones[t]) * self.GAMMA * self.GAE_LAMBDA * advantages[t + 1]

        targets = advantages + values

        # normalize advantages
        advantages -= tf.reduce_mean(advantages)
        advantages /= (tf.math.reduce_std(advantages) + 1e-8)
        advantages = tf.cast(advantages, dtype=tf.float32)

        return advantages, targets

    def learn(self, mem, targets):

        for _ in range(self.N_EPOCHS):
            total_loss = 0
            total_entropy_loss = 0
            total_actor_loss = 0
            total_critic_loss = 0

            states, actions, old_probs, values, rewards, dones, advantages, batches = mem.generate_batches(
            )

            values = tf.cast(values, dtype=tf.float32)

            for batch in batches:
                with tf.GradientTape() as tape:
                    dist, critic_value = self.agent(states[batch])

                    probs = tf.squeeze(dist.log_prob(actions[batch]))
                    critic_value = tf.squeeze(critic_value)
                    entropy = tf.reduce_mean(dist.entropy())

                    r_theta = tf.math.exp(
                        probs - tf.squeeze(tf.gather(old_probs, batch)))

                    # policy clipping
                    policy_obj = r_theta * tf.gather(advantages, batch)

                    clipped_r_theta = tf.clip_by_value(
                        r_theta, 1 - self.EPS, 1 + self.EPS) * tf.gather(
                            advantages, batch)

                    # compute losses
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(policy_obj, clipped_r_theta))

                    critic_loss = tf.reduce_mean(
                        tf.square(tf.gather(targets, batch) - critic_value))

                    loss = actor_loss + self.C1 * critic_loss + self.C2 * entropy

                    total_loss += loss
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
                    total_entropy_loss += entropy

                grads = tape.gradient(loss, self.agent.trainable_variables)
                self.opt.apply_gradients(
                    zip(grads, self.agent.trainable_variables))

                self.step += 1

        self.loss_hist.append(total_loss.numpy())
        self.actor_loss_hist.append(total_actor_loss.numpy())
        self.critic_loss_hist.append(total_critic_loss.numpy())
        self.entropy_hist.append(total_entropy_loss.numpy())

        if self.decay_lr and self.epoch % 50 == 0 and self.epoch > 0:
            # self.opt.learning_rate = self.lr_schedule(self.step).numpy()
            # self.opt.learning_rate = self.opt.learning_rate.numpy() * 0.995
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
            probs = []
            self.epoch = epoch
            self.training = True

            for t in range(self.TIMESTEPS):

                if self.render:
                    self.env.render()

                action, value, prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(np.array(action))
                episode_reward += reward

                actions.append(action)
                values.append(value.numpy())
                rewards.append(reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(done)
                probs.append(prob)

                if done:
                    state = env.reset()
                    step = 0
                    self.episode_reward_hist.append(episode_reward)
                    episode_reward = 0
                    episode += 1

                else:
                    step += 1
                    state = next_state

            _, next_values = self.agent(np.array(next_states))

            advantages, targets = self.compute_advantages(
                values, tf.squeeze(next_values), rewards, dones)

            mem.store_memory(states, actions, probs, values, rewards, dones,
                             advantages)
            self.learn(mem, targets)

            mem.clear_memory()

            if epoch % self.print_frequency == 0 and len(
                    self.episode_reward_hist) > 0:
                mean_reward = np.mean(self.episode_reward_hist[-50:])
                print(
                    f"Epoch: {epoch}\tLoss: {self.loss_hist[-1]:.2f}\tReward: {self.episode_reward_hist[-1]:.2f}\tMean reward: {mean_reward:.2f}\tLearning rate: {self.opt.learning_rate.numpy():.3e}"
                )

            if epoch % 20 == 0 and epoch > 50:
                _, test_reward = self.test(10)

                if test_reward > self.reward_threshold:
                    print(f"Solved in {epoch} epochs")
                    break

    def plot_training(self):

        window = 10
        plot_alpha = 0.7
        blue = "#4184f3"
        style = ':'

        # loss and reward
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_hist,
                 blue,
                 linestyle=style,
                 label="Loss",
                 alpha=plot_alpha)
        plt.plot(Series(self.loss_hist).rolling(window).mean(),
                 blue,
                 label=f"Rolling mean, {window}")

        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_reward_hist,
                 blue,
                 linestyle=style,
                 label="Reward",
                 alpha=plot_alpha)
        plt.plot(Series(self.episode_reward_hist).rolling(window).mean(),
                 blue,
                 label=f"Rolling mean, {window}")
        plt.ylabel("Episode reward")
        plt.xlabel("Training episode")

        plt.savefig("training_loss_reward.png")

        # diagnostics
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        plt.plot(self.actor_loss_hist,
                 blue,
                 linestyle=style,
                 label="Loss",
                 alpha=plot_alpha)
        plt.plot(Series(self.actor_loss_hist).rolling(window).mean(),
                 blue,
                 label=f"Rolling mean, {window}")
        plt.ylabel("Actor loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 2)
        plt.plot(self.critic_loss_hist,
                 blue,
                 linestyle=style,
                 label="Loss",
                 alpha=plot_alpha)
        plt.plot(Series(self.critic_loss_hist).rolling(window).mean(),
                 blue,
                 label=f"Rolling mean, {window}")
        plt.ylabel("Critic loss")
        plt.xlabel("Epochs")

        plt.subplot(3, 1, 3)
        plt.plot(self.entropy_hist,
                 blue,
                 linestyle=style,
                 label="Loss",
                 alpha=plot_alpha)
        plt.plot(Series(self.entropy_hist).rolling(window).mean(),
                 blue,
                 label=f"Rolling mean, {window}")
        plt.ylabel("Entropy loss")
        plt.xlabel("Epochs")

        plt.savefig("training_loss_diag.png")

        plt.show()
        print()

    def save_ppo_toml(self, path):
        with open(f"{path}ppo.toml", "w") as toml_file:
            toml.dump(self.__dict__, toml_file)
        with open(f"{path}ppo_actor.toml", "w") as toml_file:
            toml.dump(json.loads(self.agent.actor.to_json()), toml_file)
        with open(f"{path}ppo_critic.toml", "w") as toml_file:
            toml.dump(json.loads(self.agent.actor.to_json()), toml_file)

    def save(self):
        dt = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = f"./saved_ppo_model_{dt}/"
        self.agent.actor.save_weights(f"{path}/actor")
        self.agent.critic.save_weights(f"{path}/critic")
        print("Model saved")

        self.save_ppo_toml(path)

        return path

    def load(self, PATH):
        self.agent.actor.load_weights(f"{PATH}/actor")
        self.agent.critic.load_weights(f"{PATH}/critic")
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
                action, value, logits = self.select_action(state)

                new_state, reward, done, info = self.env.step(action)
                mem.store_memory([state], [action], [logits], [value], [reward],
                                 [done], [None])

                if render:
                    img = plt.imshow(self.env.render('rgb_array'))
                    plt.grid()
                    frames.append([img])

                state = new_state
                episode_reward += reward
                step += 1

            print(
                f"Episode {episode} lasted {step} steps. Reward: {episode_reward}"
            )
            test_episode_reward_hist.append(episode_reward)

        if render:
            an = animation.ArtistAnimation(fig,
                                           frames,
                                           interval=100,
                                           repeat_delay=1000,
                                           blit=True)
            writergif = animation.PillowWriter(fps=30)
            an.save("animation.gif", writer=writergif)
        return mem, np.mean(test_episode_reward_hist)


if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('MountainCarContinuous-v0')
    # env = ContinuousCartPoleEnv()

    agent = Agent()
    ppo = PPO(env, agent)

    ppo.train()

    ppo.plot_training()
    path = ppo.save()

    # ppo.load("./saved_ppo_model_2021-09-15T11-57-36")

    ppo.test(10, render=False)
    # ppo.test(1, render=True)
