#!/usr/bin/env python
# coding: utf-8

import gym
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

# env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')


class ActorCritic(tf.keras.Model):

    def __init__(self, env, gamma=0.99, training=True):
        super(ActorCritic, self).__init__()

        # self.inputs = layers.Input(shape=(8,))
        self.fc1 = layers.Dense(50, activation="relu", name="fc1")
        self.fc2 = layers.Dense(50, activation="relu", name="fc2")
        # self.fc3 = layers.Dense(128, activation="relu", name="fc3")
        # self.fc4 = layers.Dense(128, activation="relu", name="fc4")
        self.actor = layers.Dense(env.action_space.n,
                                  activation="linear",
                                  name="actor_out")
        self.critic = layers.Dense(1, activation="linear", name="critic_out")

        self.env = env
        self.training = training
        self.gamma = gamma

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                            beta_1=0.9,
                                            beta_2=0.999)

    def call(self, inputs):
        x1 = self.fc1(inputs)
        x1 = self.fc2(x1)
        
        actions = self.actor(x1)
        # x2 = self.fc3(inputs)
        # x2 = self.fc4(x2)
        value = self.critic(x1)

        return actions, value

    def select_action(self, inputs):

        action_logits, value = self.call(tf.expand_dims(inputs, axis=0))
        action_probs = tf.nn.softmax(action_logits)

        if self.training:
            action = tf.random.categorical(action_logits, 1)[0, 0].numpy()

        else:
            action = tf.math.argmax(tf.squeeze(action_probs)).numpy()

        return action, tf.math.log(tf.squeeze(action_probs)[action]), value

    def train_step(self, state):

        action, log_action, value = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)

        return action, log_action, value, next_state, reward, done

    def learn(self, reward, log_action, v_pred, done, eps=1e-9):

        v_target = reward - self.gamma * v_pred * (1 - int(done))

        # compute loss
        actor_loss = tf.squeeze(-log_action * (v_target - v_pred))
        critic_loss = tf.keras.losses.Huber()(v_target, v_pred)
        # critic_loss = tf.squeeze(v_target - v_pred)**2
        loss = actor_loss + critic_loss  #total loss

        # update policy
        grads = self.tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss

    def train(self, episodes, max_steps=1000):
        episode_reward_hist = []
        loss_hist = []

        for episode in range(episodes):

            step = 0
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_loss = 0

            while not done and step < max_steps:

                with tf.GradientTape() as self.tape:

                    action, log_action, value, next_state, reward, done = self.train_step(
                        state)
                    loss = self.learn(reward, log_action, value, done)

                    # update
                    episode_reward += reward
                    episode_loss += loss
                    step += 1
                    state = next_state

            loss_hist.append(episode_loss)
            episode_reward_hist.append(episode_reward)

            if episode % 10 == 0:
                print(
                    f"Episode:{episode}, Loss: {episode_loss.numpy():.2f}, Ep reward:{episode_reward:.2f}"
                )
        return loss_hist, episode_reward_hist


ac = ActorCritic(env)
loss_hist, episode_reward_hist = ac.train(500, max_steps=500)

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

an = animation.ArtistAnimation(fig,
                               images,
                               interval=100,
                               repeat_delay=1000,
                               blit=True)
rc('animation', html='jshtml')
an

writergif = animation.PillowWriter(fps=30)
an.save(image_file, writer=writergif)
