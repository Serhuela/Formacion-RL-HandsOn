import torch
import torch.nn as nn
import numpy
import torch.optim as optim
from tensorboardX import SummaryWriter

print("Starting...")
print("In order to track learning, please execute:")
print("tensorboard --logdir src/runs --host localhost")
batch_size = 50000
increment = float(batch_size)/50000
limit = 100
precision = 0.03
initial_error_print = limit*limit
neural_net = nn.Sequential(
    nn.Linear(1,5),
    nn.ReLU(),
    nn.Linear(5,20),
    nn.ReLU(),
    nn.Linear(20,60),
    nn.ReLU(),
    nn.Linear(60,20),
    nn.ReLU(),
    nn.Linear(20,5),
    nn.ReLU(),
    nn.Linear(5,1)
)
loss = nn.MSELoss()
optimizer = optim.Adagrad(neural_net.parameters())
writer = SummaryWriter()

step = 0
while True:
    optimizer.zero_grad()
    np_input = numpy.random.normal(scale=limit, loc=0, size=(batch_size,1))
    np_target = numpy.abs(np_input)
    t_input = torch.FloatTensor(np_input)
    t_target = torch.FloatTensor(np_target)
    t_output = neural_net(t_input)
    error = loss(t_target, t_output)
    error.backward()
    writer.add_scalar("Error", error.item(), step)
    step += increment
    optimizer.step()
    if error.item() < initial_error_print:
        print("%15.5f ---> %15.5f" % (initial_error_print,error.item()))
        initial_error_print = initial_error_print * 0.99
    if error.item() < precision:
        writer.close()
        break
print("Exiting...")
