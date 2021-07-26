#!/usr/bin/env python
# coding: utf-8

import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import numpy as np
import logging
tf.get_logger().setLevel(logging.INFO)
# get_ipython().run_line_magic('matplotlib', 'inline')

env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')
env.reset()
# plt.imshow(env.render(mode="rgb_array"));


class ActorCritic(tf.keras.Model):

    def __init__(self, env, gamma=0.99, training=True):
        super(ActorCritic, self).__init__()

        # self.inputs = layers.Input(shape=(8,))
        self.fc1 = layers.Dense(128, activation="relu", name="fc1")
        # self.fc2 = layers.Dense(128, activation="relu", name="fc2")
        # self.fc3 = layers.Dense(128, activation="relu", name="fc3")
        # self.fc4 = layers.Dense(128, activation="relu", name="fc4")
        self.actor = layers.Dense(env.action_space.n,
                                  activation="linear",
                                  name="actor_out")
        self.critic = layers.Dense(1, activation="linear", name="critic_out")

        self.env = env
        self.training = training
        self.gamma = gamma

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)



    def call(self, inputs):
        x1 = self.fc1(inputs)
        # x1 = self.fc2(x1)
        actions = self.actor(x1)
        
        # x2 = self.fc3(inputs)
        # x2 = self.fc4(x2)
        value = self.critic(x1)
        
        
        return actions, value

    def select_action(self, inputs):

        action_logits, value = self.call(tf.expand_dims(inputs, axis=0))
        action_probs = tf.nn.softmax(action_logits)

        if self.training:
            action = tf.random.categorical(action_logits, 1)[0,0].numpy()

        else:
            action = tf.math.argmax(tf.squeeze(action_probs)).numpy()

        return action, tf.math.log(tf.squeeze(action_probs)[action]), value

    def train_step(self, state):

        action, log_action, value = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)

        return action, log_action, value, next_state, reward, done

    def learn(self, rewards, log_actions, v_preds, v_targets):
        
        # update value estimates
        vt = tf.constant(0.0)
        vt = rewards[-1]
        v_targets = v_targets.write(len(rewards)-1, vt)
        for t in reversed(range(len(rewards)-1)):
            vt = rewards[t] + (self.gamma * vt)
            v_targets = v_targets.write(t, vt)

        v_targets = v_targets.stack()
        v_targets = (v_targets - tf.math.reduce_mean(v_targets)) / (tf.math.reduce_std(v_targets) + 1e-9)

        # compute loss
        actor_loss = tf.math.reduce_sum(-log_actions * (v_targets - v_preds))
        critic_loss = tf.keras.losses.Huber()(v_targets, v_preds) * len(rewards)
        loss = actor_loss + critic_loss

        # update policy
        grads = self.tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, episodes, max_steps=500):
        episode_reward_hist = np.zeros(episodes)
        loss_hist = []

        for episode in range(episodes):

            step = 0
            state = self.env.reset()
            done = False
            episode_reward = 0

            with tf.GradientTape() as self.tape:

                #initialize arrays
                rewards = tf.TensorArray(dtype=tf.float32,
                                         size=0,
                                         dynamic_size=True)
                log_actions = tf.TensorArray(dtype=tf.float32,
                                             size=0,
                                             dynamic_size=True)
                v_preds = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)
                v_targets = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

                while not done:# and step < max_steps:

                    action, log_action, value, next_state, reward, done = self.train_step(
                        state)

                    rewards = rewards.write(step, reward)
                    log_actions = log_actions.write(step, log_action)
                    v_preds = v_preds.write(step, value)

                    episode_reward += reward
                    step += 1
                    state = next_state

                rewards = rewards.stack()
                log_actions = log_actions.stack()
                v_preds = tf.squeeze(v_preds.stack())

                loss = self.learn(rewards, log_actions, v_preds, v_targets)

            loss_hist.append(loss)
            episode_reward_hist[episode] = episode_reward

            if episode % 10 == 0:
                print(
                    f"Episode:{episode}, Loss: {loss.numpy():.2f}, Ep reward:{episode_reward:.2f}")
        return loss_hist, episode_reward_hist


ac = ActorCritic(env)
loss_hist, episode_reward_hist = ac.train(500)

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(loss_hist)
plt.grid()
plt.title("Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")

plt.subplot(2, 1, 2)
plt.plot(episode_reward_hist)
plt.ylabel("Reward")
plt.xlabel("Episodes")
plt.title("Episode Reward")
plt.grid()
plt.tight_layout()
plt.show()

from matplotlib import animation, rc

fig = plt.figure()
images = []


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):

    state = tf.constant(env.reset(), dtype=tf.float32)
    for i in range(1, max_steps + 1):
        action, _, _ = ac.select_action(state)

        state, _, done, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)

        # Render screen every 2 steps
        if i % 2 == 0:
            screen = plt.imshow(env.render(mode='rgb_array'))
            images.append([screen])

        if done:
            break

    return images


# Save GIF image
ac.training = False
images = render_episode(env, ac, 500)
image_file = 'actor_critic.gif'

an = animation.ArtistAnimation(fig, images, interval=100, repeat_delay=1000, blit=True)
rc('animation', html='jshtml')
an

writergif = animation.PillowWriter(fps=30)
an.save(image_file, writer=writergif)
