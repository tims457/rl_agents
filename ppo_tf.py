#!/usr/bin/env python
# coding: utf-8

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import numpy as np
import copy
from datetime import datetime

tf.random.set_seed(0)
np.random.seed(0)


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
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(
            self.logits), np.array(self.vals), np.array(self.rewards), np.array(
                self.dones), np.array(self.advantages), batches

    def store_memory(self, state, action, logits, vals, reward, done,
                     advantage):
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


class Agent(tf.keras.Model):

    def __init__(self):
        super(Agent, self).__init__()

        #         self.fc1 = layers.Dense(512, activation="relu",
        #                                 kernel_initializer=tf.keras.initializers.he_normal(), name="fc1")
        #         self.fc2 = layers.Dense(256, activation="relu",
        #                                 kernel_initializer=tf.keras.initializers.he_normal(), name="fc2")
        #         self.fc3 = layers.Dense(64, activation="relu",
        #                                 kernel_initializer=tf.keras.initializers.he_normal(), name="fc3")

        #         self.actor = layers.Dense(env.action_space.n,
        #                                   kernel_initializer=tf.keras.initializers.he_normal(),
        #                                   activation="linear",
        #                                   name="actor_out")

        #         self.critic = layers.Dense(1, activation="linear", kernel_initializer=tf.keras.initializers.he_normal(), name="critic_out")

        self.actor = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(512,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(256,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(64,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(env.action_space.n, activation="linear"),
        ])

        self.critic = tf.keras.Sequential([
            layers.InputLayer(input_shape=(env.observation_space.shape[0],)),
            layers.Dense(512,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(256,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(64,
                         activation="relu",
                         kernel_initializer=tf.keras.initializers.he_uniform()),
            layers.Dense(1, activation="linear")
        ])

    def call(self, inputs):

        policy_logits = self.actor(inputs)
        dist = tfd.Categorical(logits=policy_logits)
        value = self.critic(inputs)

        #         x = self.fc1(inputs)
        #         x = self.fc2(x)
        #         x = self.fc3(x)
        #         policy_logits = self.actor(x)
        #         dist = tfd.Categorical(logits=policy_logits)
        #         value = self.critic(x)

        return dist, value, policy_logits


class PPO():

    def __init__(self, env, agent, training=True):
        self.env = env
        self.agent = agent
        self.training = training

        self.EPOCHS = 200
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.EPS = 0.1

        self.TIMESTEPS = 1000
        self.BATCH_SIZE = 64
        self.N_EPOCHS = 10

        self.C1 = 0.5 #0.5  #critic loss coefficient
        self.C2 = 0.001  #entropy coefficient

        self.opt = tf.keras.optimizers.Adam(learning_rate=2.5e-4)

        self.print_frequency = 1
        self.render = False

    def select_action(self, state, test=False):
        dist, value, policy_logits = self.agent(tf.expand_dims(state, axis=0))

        if self.training:
            action = tf.squeeze(dist.sample(1)).numpy()
        else:
            action = np.argmax(tf.squeeze(dist.probs_parameter()).numpy())

        return action, tf.squeeze(value), tf.squeeze(policy_logits)

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
        
        # normalize #? should this be before or after the targets?
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

            states, actions, logits_old, values, rewards, dones, advantages, batches = mem.generate_batches(
            )

            action_idx = tf.stack([tf.range(0, len(rewards)), actions], axis=1)

            old_probs = tf.nn.softmax(tf.squeeze(logits_old))
            old_probs = tf.gather_nd(old_probs, action_idx)

            values = tf.cast(values, dtype=tf.float32)

            for batch in batches:
                with tf.GradientTape() as tape:
                    dist, critic_value, logits = self.agent(states[batch])
                    probs = tf.nn.softmax(logits)

                    action_idx_batch = tf.stack(
                        [tf.range(0, len(probs)), actions[batch]], axis=1)

                    probs = tf.gather_nd(probs, action_idx_batch)

                    critic_value = tf.squeeze(critic_value)

                    entropy = tf.reduce_mean(dist.entropy())

                    # distribution ratio
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

        self.loss_hist.append(total_loss)
        self.actor_loss_hist.append(total_actor_loss)
        self.critic_loss_hist.append(total_critic_loss)
        self.entropy_hist.append(total_entropy_loss)

    def train(self):

        self.training = True
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

            for t in range(self.TIMESTEPS):

                if self.render:
                    self.env.render()

                action, value, pls = self.select_action(state, agent)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                actions.append(action)
                values.append(value.numpy())
                rewards.append(reward)
                states.append(state)
                next_states.append(next_state)
                dones.append(done)
                policy_logits.append(pls)

                if done:
                    state = env.reset()
                    step = 0
                    self.episode_reward_hist.append(episode_reward)
                    episode_reward = 0
                    episode += 1

                else:
                    step += 1
                    state = next_state

            _, next_values, _ = self.agent(np.array(next_states))

            advantages, targets = self.compute_advantages(
                values, tf.squeeze(next_values), rewards, dones)

            mem.store_memory(states, actions, policy_logits, values, rewards,
                             dones, advantages)
            self.learn(mem, targets)

            mem.clear_memory()

            if epoch % self.print_frequency == 0:
                print(
                    f"Epoch: {epoch}\tLoss: {self.loss_hist[-1]:.2f}\tReward: {self.episode_reward_hist[-1]:.2f}\tMean reward: {np.mean(self.episode_reward_hist[-50:]):.2f}"
                )

    def plot_training(self):
        # loss and reward
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_hist)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(self.episode_reward_hist)
        plt.ylabel("Episode reward")
        plt.xlabel("Training episode")
        plt.grid()

        plt.savefig("training_loss_reward.png")

        # diagnostics
        plt.figure(figsize=(10, 12))
        plt.subplot(3, 1, 1)
        plt.plot(self.actor_loss_hist)
        plt.ylabel("Actor loss")
        plt.xlabel("Epochs")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(self.critic_loss_hist)
        plt.ylabel("Critic loss")
        plt.xlabel("Epochs")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(self.entropy_hist)
        plt.ylabel("Entropy loss")
        plt.xlabel("Epochs")
        plt.grid()

        plt.savefig("training_loss_diag.png")

        plt.show()

    def save(self):
        dt = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = f"./saved_ppo_model_{dt}/"
        self.agent.actor.save_weights(f"{path}/actor")
        self.agent.critic.save_weights(f"{path}/critic")
        print("Model saved")
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

        if render:

            fig = plt.figure()

        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0

            while not done:
                action, value, logits = self.select_action(state, test=True)

                new_state, reward, done, info = self.env.step(action)
                mem.store_memory([state], [action], [logits], [value], [reward],
                                 [done], [None])

                if render:
                    img = plt.imshow(self.env.render('rgb_array'))
                    frames.append([img])

                state = new_state
                episode_reward += reward
                step += 1

            print(
                f"Episode {episode} lasted {step} steps. Reward: {episode_reward}"
            )

        if render:
            an = animation.ArtistAnimation(fig,
                                           frames,
                                           interval=100,
                                           repeat_delay=1000,
                                           blit=True)
            writergif = animation.PillowWriter(fps=30)
            an.save("animation.gif", writer=writergif)
        return mem


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('CartPole-v1')
    agent = Agent()
    ppo = PPO(env, agent)

    ppo.train()
    ppo.plot_training()
    path = ppo.save()

    # ppo.load("./saved_ppo_model_2021-08-28T14-02-10")

    ppo.test(10, render=False)
