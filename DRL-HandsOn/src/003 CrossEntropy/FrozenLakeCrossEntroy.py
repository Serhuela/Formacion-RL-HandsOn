import gym
import math
import torch
import numpy as np
from torch import nn
from torch import optim
from collections import namedtuple
from tensorboardX import SummaryWriter


BATCH_SIZE = 1000
ENV_NAME = "FrozenLake-v0"
env = gym.make(ENV_NAME)
env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
HIDDEN_LAYER_SIZE = 200
PERCENTILE = 70
SHOW_SOME = False
DISCOUNT_FACTOR = 0.9
DEVICE = torch.device(type="cuda")

net = nn.Sequential(
    nn.Linear(env.observation_space.n, HIDDEN_LAYER_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_LAYER_SIZE, env.action_space.n)
).to(DEVICE)

sm = nn.Softmax(dim=1)
optimizer = optim.Adagrad(net.parameters())
loss = nn.CrossEntropyLoss()

writer = SummaryWriter(comment=("-" + ENV_NAME))

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


class OneHotEncoder:
    def __init__(self,
                 width):
        self.width = width

    def encode(self, code):
        a = np.zeros(self.width)
        a[code] = 1
        return a


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    onehot = OneHotEncoder(env.observation_space.n)

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward <= reward_bound:
            continue
        train_obs.extend(map(lambda step: onehot.encode(step.observation), example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs).to(DEVICE)
    train_act_v = torch.LongTensor(train_act).to(DEVICE)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    min_reward = 0.0
    iter_no = 0
    onehot = OneHotEncoder(env.observation_space.n)
    while min_reward < 10000:
        batch_history = []
        show = SHOW_SOME
        for _ in range(BATCH_SIZE):
            episode_acc_reward = 0.0
            episode_history = []
            obs = env.reset()
            if show:
                env.render()
            done = False
            while not done:
                t_obs = torch.FloatTensor([onehot.encode(obs)]).to(DEVICE)
                t_act_prob = sm(net(t_obs))
                act_prob = t_act_prob.to(torch.device("cpu")).data.numpy()[0]
                act = np.random.choice(len(act_prob), p=act_prob)
                new_obs, reward, done, _ = env.step(act)
                if show:
                    env.render()
                episode_history.append(EpisodeStep(observation=obs, action=act))
                obs = new_obs
                episode_acc_reward += reward
            batch_history.append(Episode(reward=episode_acc_reward*math.pow(DISCOUNT_FACTOR,episode_history.__len__()),steps=episode_history))
            if show:
                env.close()
                show = False
        batch_rewards = list(map(lambda x: x.reward, batch_history))
        min_reward = min(batch_rewards)
        print("Min reward was: %7.2f." % min_reward)
        print("Max reward was: %7.2f." % max(batch_rewards))
        # Train
        train_obs_v, train_act_v, reward_bound, reward_mean = filter_batch(batch_history,PERCENTILE)
        print(len(list(filter(lambda x: x > reward_bound, batch_rewards))))
        print("Next reward bound is: %7.2f." % reward_bound)
        optimizer.zero_grad()
        act_score = net(train_obs_v)
        error = loss(act_score, train_act_v)
        error.backward()
        optimizer.step()
        # End train
        writer.add_scalar("error", error.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        iter_no += 1
    writer.close()
