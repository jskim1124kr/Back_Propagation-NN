from NN import NeuralNet
from data_load import load
import matplotlib.pyplot as plt
import numpy as np

# Data load
x,y = load()
shuffle_indices = np.random.permutation((len(y)))

x = x[shuffle_indices]
y = y[shuffle_indices]
train_len = int(len(x) * 0.6)
val_len = int(len(x) * 0.2)
test_len = int(len(x) * 0.2)
x_train = x[:train_len]
y_train = y[:train_len]
x_val = x[train_len:train_len+val_len]
y_val = y[train_len:train_len+val_len]
x_test = x[train_len+val_len:]
y_test = y[train_len+val_len:]

print("x_train len : [ " + str(len(x_train)) + " ]")
print("x_val len  : [ " + str(len(x_val)) + " ]")
print("x_test len : [ " + str(len(x_test)) + " ]")




train_size = x_train.shape[0]
batch_size = 20
iters_num = 10000
learning_rate = 0.01

train_loss_list = []
val_loss_list = []
iter_per_epoch = max(train_size / batch_size,1)





network = NeuralNet(input_size=4, hidden_size=5, output_size=3)
print("Training...")
for _ in range(iters_num):
    grad = network.propagation(x_train, y_train)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if _ % 1000 == 0:
        train_cost = network.loss(x_train, y_train)
        val_cost = network.loss(x_val, y_val)
        train_loss_list.append(train_cost)
        val_loss_list.append(val_cost)

    if _ % 1000 == 0:
        train_cost2 = network.loss(x_train, y_train)
        val_cost2 = network.loss(x_val, y_val)
        print("Step : [ " + str(_) + " ] : " + "train_Cost : [ " + str(train_cost2) + " ]")
        print("Step : [ " + str(_) + " ] : " + "validatin_Cost : [ " + str(val_cost2) + " ]")


x_axis = []
for i in range(0,10000,1000):
    x_axis.append(i)

plt.plot(x_axis, train_loss_list, 'b')
plt.plot(x_axis, val_loss_list, 'g')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()



