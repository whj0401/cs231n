from cs231n.data_utils import load_CIFAR10
import numpy as np
from cs231n.classifiers.neural_net import *
from cs231n.vis_utils import visualize_grid
from threading import Lock
from threading import Thread
import threading
import copy
import time

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'E:\\learning_meterial\\cs231n\\assignment1\\cs231n\\datasets\\cifar-10-batches-py'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
# try:
#    del X_train, y_train
#    del X_test, y_test
#    print('Clear previously loaded data.')
# except:
#    pass

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
# net = TwoLayerNet(input_size, hidden_size, num_classes)
#
# # Train the network
# stats = net.train(X_train, y_train, X_val, y_val,
#             num_iters=1000, batch_size=200,
#             learning_rate=1e-4, learning_rate_decay=0.95,
#             reg=0.25, verbose=True)
#
# # Predict on the validation set
# val_acc = (net.predict(X_val) == y_val).mean()
# print('Validation accuracy: ', val_acc)


# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


# show_net_weights(net)


best_net_lock = Lock()
best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
best_acc = 0.0
tmp = 5e-5
learning_rates = [tmp * (2**x) for x in range(5)]
#learning_rate_decays = [1 - 1 / (x+2) for x in range(10)]
learning_rate_decays = [0.95]
tmp = 1e-5
regs = [tmp * (10**x) for x in range(5)]
hidden_sizes = [30, 50, 70, 100, 250, 1000]
task_count = 0
num_tasks = len(learning_rates) * len(learning_rate_decays) * len(hidden_sizes) * len(regs)


def train_hyperparameters(hs, lr, lrd, r):
    net = TwoLayerNet(input_size, hs, num_classes)
    global task_count
    global best_net
    global best_acc
    stats = net.train(X_train, y_train, X_val, y_val,
        num_iters=1000, batch_size=200,
        learning_rate=lr, learning_rate_decay=lrd,
        reg=r, verbose=False)
    tmp_acc = (net.predict(X_val) == y_val).mean()
    print("try hidden_size=%d, learning_rate=%e, learning_rate_decay=%e, reg=%e, accuracy=%e"
                      % (hs, lr, lrd, r, tmp_acc))
    best_net_lock.acquire(blocking=True, timeout=-1)
    if tmp_acc > best_acc:
        best_net = copy.deepcopy(net)
        best_acc = tmp_acc
        print("contemporary best_net is hidden_size=%d, learning_rate=%e, learning_rate_decay=%e, reg=%e, accuracy=%e"
                      % (hs, lr, lrd, r, tmp_acc))
    task_count += 1
    best_net_lock.release()


# class TrainThread(Thread):
#     def __init__(self, hs, lr, lrd, r):
#         self.__hs = hs
#         self.__lr = lr
#         self.__lrd = lrd
#         self.__r = r
#
#
#     def run(self):
#         train_hyperparameters(self.__hs, self.__lr, self.__lrd, self.__r)


train_lock = Lock()
train_lock.acquire(blocking=True, timeout=-1)
for learning_rate in learning_rates:
    for learning_rate_decay in learning_rate_decays:
        for reg in regs:
            for hidden_size in hidden_sizes:
                while threading.activeCount() > 31:
                    time.sleep(1)
                try:
                    t = Thread(target=train_hyperparameters, args=(hidden_size, learning_rate, learning_rate_decay, reg))
                    t.start()
                except:
                    task_count += 1
                    print("try hidden_size=%d, learning_rate=%e, learning_rate_decay=%e, reg=%e failed"
                          % (hidden_size, learning_rate, learning_rate_decay, reg))
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

while task_count < num_tasks:
    time.sleep(10)

# visualize the weights of the best network
show_net_weights(best_net)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
