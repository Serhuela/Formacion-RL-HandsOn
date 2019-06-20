import gym
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch import optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

# Parameters / constant variables
IMG_SIZE = 128
ENV_NAME = "Breakout-v0"
BATCH_SIZE = 16
LATENT_VECTOR_SIZE = 100

DIS_FILTERS = 64
GEN_FILTERS = 64
DEVICE = torch.device(type="cuda")

FACTOR = 1.26

class BatchProvider:
    def __init__(self,
                 img_size,
                 env_name):
        self.img_size = img_size
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env.reset()

    def get_batch(self,batch_size):
        batch = []
        while len(batch) < batch_size:
            obs, _, done, _ = self.env.step(action=self.env.action_space.sample())
            if done:
                self.env.reset()
            batch.append(np.moveaxis(cv2.resize(obs, (self.img_size,self.img_size)), 2, 0))
        return np.array(batch)


if __name__ == "__main__":
    print("Starting...")
    print("In order to track learning, please execute:")
    print("tensorboard --logdir runs --host localhost")

    log = gym.logger
    log.set_level(level=gym.logger.INFO)
    log.info("Iter  gen_loss=, dis_loss=")

    discriminator_net = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=DIS_FILTERS,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(DIS_FILTERS),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=DIS_FILTERS,
            out_channels=DIS_FILTERS * 2,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(DIS_FILTERS * 2),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=DIS_FILTERS * 2,
            out_channels=DIS_FILTERS * 4,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(DIS_FILTERS * 4),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=DIS_FILTERS * 4,
            out_channels=DIS_FILTERS * 8,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(DIS_FILTERS * 8),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=DIS_FILTERS * 8,
            out_channels=DIS_FILTERS * 16,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(DIS_FILTERS * 16),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=DIS_FILTERS * 16,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0
        ),
        nn.Sigmoid()
    ).to(DEVICE)

    generator_net = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=LATENT_VECTOR_SIZE,
            out_channels=GEN_FILTERS * 16,
            kernel_size=4,
            stride=1,
            padding=0
        ),
        nn.BatchNorm2d(GEN_FILTERS * 16),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=GEN_FILTERS * 16,
            out_channels=GEN_FILTERS * 8,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(GEN_FILTERS * 8),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=GEN_FILTERS * 8,
            out_channels=GEN_FILTERS * 4,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(GEN_FILTERS * 4),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=GEN_FILTERS * 4,
            out_channels=GEN_FILTERS * 2,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(GEN_FILTERS * 2),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=GEN_FILTERS * 2,
            out_channels=GEN_FILTERS,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.BatchNorm2d(GEN_FILTERS),
        nn.ReLU(),
        nn.ConvTranspose2d(
            in_channels=GEN_FILTERS,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        ),
        nn.Tanh()
    ).to(DEVICE)
    batch_provider = BatchProvider(img_size=IMG_SIZE,env_name=ENV_NAME)

    true_labels_v = torch.ones(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    fake_labels_v = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    gen_optimizer = optim.Adagrad(generator_net.parameters())
    dis_optimizer = optim.Adagrad(discriminator_net.parameters())

    dis_loss = nn.BCELoss()
    gen_loss = nn.BCELoss()

    writer = SummaryWriter()

    step = -1
    should_write = FACTOR

    while True:
        step += 1
        # Generate True Batch
        true_batch = batch_provider.get_batch(batch_size=BATCH_SIZE)
        normalized_true_batch = (true_batch.astype("float32")-127.5)/128
        # Generate Latent Vector
        fake_seed_batch = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device=DEVICE)

        # Generate Fake Batch
        fake_img_batch = generator_net(fake_seed_batch)

        # Train Discriminator
        dis_optimizer.zero_grad()
        img_torch_batch = torch.tensor(normalized_true_batch).to(device=DEVICE)
        dis_error = dis_loss(discriminator_net(img_torch_batch), true_labels_v) +\
            dis_loss(discriminator_net(fake_img_batch.detach()), fake_labels_v)
        dis_error.backward()
        dis_optimizer.step()

        # Train Generator
        gen_optimizer.zero_grad()
        fake_values = discriminator_net(fake_img_batch)
        gen_error = gen_loss(fake_values, true_labels_v)
        gen_error.backward()
        gen_optimizer.step()

        if step > should_write:
            should_write *= FACTOR
            writer.add_scalar("Discriminator", dis_error.item(), step)
            writer.add_scalar("Generator", gen_error.item(), step)
            writer.add_image("fake", vutils.make_grid(fake_img_batch.data[:IMG_SIZE], normalize=True), step)

        print("Real images shown: %8d" % (step * BATCH_SIZE))
    writer.close()
