import gym
import math
import torch
import numpy as np
from torch import nn
from torch import optim
from operator import attrgetter
from collections import namedtuple
from tensorboardX import SummaryWriter


BATCH_SIZE = 500
ENV_NAME = "FrozenLake-v0"
MIN_REWARD = 0
HIDDEN_LAYER_SIZE = 200
SHOW_BEGINNING = False
SHOW_SOME = False
DISCOUNT_DECAY = 0.9999
TRAINING_DECAY = 0.999
AGING_DECAY = 0.99
DEVICE = torch.device(type="cuda")
NAME_MODIFIER = "-" + ENV_NAME + "-CrossEntropy-Modified"
HISTORY_FACTOR = 10

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


class HistoryEpisode:
    def __init__(self,
                 episode,
                 born_train_year):
        self.episode = episode
        self.born_train_year = born_train_year
        self.last_training_age = 0
        self.usages = 0
        self.score = episode.reward
        self.reward = episode.reward
        self.delete_points = 0

    def increase_usages(self,
                        train_year,
                        training_decay):
        self.last_training_age = train_year
        self.usages += 1
        self.score *= training_decay

    def increase_delete_points(self):
        self.delete_points += 1

    def get_older(self, aging_decay):
        self.score *= aging_decay

    def get_delete_points(self):
        return self.delete_points

    def get_episode(self):
        return self.episode

    def get_score(self):
        return self.score

    def get_reward(self):
        return self.episode.reward


class HistoryManager:
    def __init__(self,
                 minimum_reward,
                 training_decay,
                 aging_decay,
                 history_factor):
        self.History = []
        self.train_steps = 0
        self.min_reward = minimum_reward
        self.batch_size = BATCH_SIZE
        self.training_decay = training_decay
        self.aging_decay = aging_decay
        self.history_factor = history_factor
        self.max_delete_points = self.batch_size * self.history_factor
        self.added_since_last_training = 0

    def add_episode(self, episode: Episode) -> None:
        if episode.reward > self.min_reward:
            self.History.append(HistoryEpisode(episode=episode,
                                               born_train_year=self.train_steps))
            aux = min(self.History.__len__(), self.max_delete_points)
            self.added_since_last_training += 1
            self.History = sorted(self.History, key=attrgetter("score"), reverse=True)
            self.History = list(filter(lambda x: x.get_delete_points() < self.max_delete_points, self.History))
            for x in range(aux):
                self.History[len(self.History) - 1 - x].increase_delete_points()

    def get_history_size(self):
        return self.History.__len__()

    def get_added_since_last_training(self):
        return self.added_since_last_training

    def get_batch(self):
        self.added_since_last_training = 0
        self.train_steps += 1
        self.History = sorted(self.History, key=attrgetter("score"), reverse=True)
        for x in range(self.batch_size):
            self.History[x].increase_usages(self.train_steps, self.training_decay)
        for episode in self.History:
            episode.get_older(self.aging_decay)
        to_print = list(map(lambda x: x.get_reward(), self.History[:self.batch_size]))
        return self.History[:self.batch_size]


def filter_batch(batch):
    train_obs = []
    train_act = []
    for example in batch:
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    train_obs_v = torch.FloatTensor(train_obs).to(DEVICE)
    train_act_v = torch.LongTensor(train_act).to(DEVICE)
    return train_obs_v, train_act_v


def play_discrete_discrete_episode(policy,
                                   environment,
                                   onehot,
                                   softmax,
                                   discount_factor,
                                   show):
    episode_history = []
    episode_acc_reward = 0
    obs = environment.reset()
    obs = onehot.encode(obs)

    done = False
    while not done:
        t_obs = torch.FloatTensor([obs]).to(DEVICE)
        t_act = policy(t_obs)
        t_act_prob = softmax(t_act)
        act_prob = t_act_prob.to(torch.device("cpu")).data.numpy()[0]
        act = np.random.choice(len(act_prob), p=act_prob)
        new_obs, reward, done, _ = env.step(act)
        if show:
            env.render()
        episode_history.append(EpisodeStep(observation=obs, action=act))
        obs = onehot.encode(new_obs)
        episode_acc_reward += reward
    return Episode(reward=episode_acc_reward * math.pow(discount_factor,episode_history.__len__()), steps=episode_history)


def play_continuous_discrete_episode(policy,
                                     environment,
                                     softmax,
                                     discount_factor,
                                     show):
    episode_history = []
    episode_acc_reward = 0
    obs = environment.reset()

    done = False
    while not done:
        t_obs = torch.FloatTensor([obs]).to(DEVICE)
        t_act = policy(t_obs)
        t_act_prob = softmax(t_act)
        act_prob = t_act_prob.to(torch.device("cpu")).data.numpy()[0]
        act = np.random.choice(len(act_prob), p=act_prob)
        new_obs, reward, done, _ = env.step(act)
        if show:
            env.render()
        episode_history.append(EpisodeStep(observation=obs, action=act))
        obs = new_obs
        episode_acc_reward += reward
    return Episode(reward=episode_acc_reward * math.pow(discount_factor,episode_history.__len__()), steps=episode_history)


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    env = gym.wrappers.Monitor(env, directory="mon-" + NAME_MODIFIER, force=True)
    writer = SummaryWriter(comment=NAME_MODIFIER)

    if isinstance(env.observation_space, gym.spaces.Box):
        if env.observation_space.shape.__len__() > 1:
            print("Este entorno no es del tipo para el que está preparado este fichero")
    else:
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            print("Este entorno no es del tipo para el que está preparado este fichero")
    input_size = 0
    if isinstance(env.observation_space, gym.spaces.Box):
        input_size = env.observation_space.shape[0]
    else:
        input_size = env.observation_space.n
    output_size = 0
    if isinstance(env.action_space, gym.spaces.Box):
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n

    net = nn.Sequential(
        nn.Linear(input_size, HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_LAYER_SIZE, output_size)
    ).to(DEVICE)

    print("CheckPoint")

    sm = nn.Softmax(dim=1)
    optimizer = optim.Adagrad(net.parameters())
    loss = nn.CrossEntropyLoss()

    one_hot = OneHotEncoder(input_size)

    min_reward = 0.0
    iter_no = 0

    history = HistoryManager(MIN_REWARD, TRAINING_DECAY, AGING_DECAY, HISTORY_FACTOR)
    last1000_mean = np.zeros(1000).tolist()

    while history.get_history_size() < BATCH_SIZE:
        iter_no += 1
        if isinstance(env.observation_space, gym.spaces.Box):
            episode_history = play_continuous_discrete_episode(policy=net,
                                                               environment=env,
                                                               softmax=sm,
                                                               discount_factor=DISCOUNT_DECAY,
                                                               show=SHOW_BEGINNING)
        else:
            episode_history = play_discrete_discrete_episode(policy=net,
                                                             environment=env,
                                                             onehot=one_hot,
                                                             softmax=sm,
                                                             discount_factor=DISCOUNT_DECAY,
                                                             show=SHOW_BEGINNING)
        last1000_mean[iter_no % 1000] = episode_history.reward
        history.add_episode(episode_history)
        print("History Length: %10d" % history.History.__len__())
        writer.add_scalar("reward", episode_history.reward, iter_no)

    print("Initial plays are done.")

    learning_iter = 0

    while min_reward < 10000:
        # learn
        if history.get_added_since_last_training() > 0:
            learning_iter += 1
            batch = history.get_batch()
            learning_scores = list(map(lambda x: x.get_score(), batch))
            writer.add_scalar("min_learning_score_play", min(learning_scores), iter_no)
            writer.add_scalar("max_learning_score_play", max(learning_scores), iter_no)
            writer.add_scalar("min_learning_score_learning", min(learning_scores), learning_iter)
            writer.add_scalar("max_learning_score_learning", max(learning_scores), learning_iter)
            learning_rewards = list(map(lambda x: x.get_reward(), batch))
            writer.add_scalar("min_learning_reward_play", min(learning_rewards), iter_no)
            writer.add_scalar("max_learning_reward_play", max(learning_rewards), iter_no)
            writer.add_scalar("min_learning_reward_learning", min(learning_rewards), learning_iter)
            writer.add_scalar("max_learning_reward_learning", max(learning_rewards), learning_iter)
            train_obs_v, train_act_v = filter_batch(list(map(lambda x: x.get_episode(), batch)))
            optimizer.zero_grad()
            act_score = net(train_obs_v)
            error = loss(act_score, train_act_v)
            error.backward()
            optimizer.step()
        # play
        iter_no += 1
        if isinstance(env.observation_space, gym.spaces.Box):
            episode_history = play_continuous_discrete_episode(policy=net,
                                                               environment=env,
                                                               softmax=sm,
                                                               discount_factor=DISCOUNT_DECAY,
                                                               show=SHOW_BEGINNING)
        else:
            episode_history = play_discrete_discrete_episode(policy=net,
                                                             environment=env,
                                                             onehot=one_hot,
                                                             softmax=sm,
                                                             discount_factor=DISCOUNT_DECAY,
                                                             show=SHOW_BEGINNING)
        last1000_mean[iter_no % 1000] = episode_history.reward
        print(sum(last1000_mean)/len(last1000_mean))
        history.add_episode(episode_history)
        writer.add_scalar("reward", episode_history.reward, iter_no)

    writer.close()
