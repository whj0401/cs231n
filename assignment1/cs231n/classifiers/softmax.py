import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim = X.shape
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        loss_i = np.log(np.sum(np.exp(scores))) - scores[y[i]]
        loss += loss_i
        for j in range(num_classes):
            tmp = np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i]
            if j == y[i]:
                dW[:, j] += tmp - X[i]
            else:
                dW[:, j] += tmp
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim = X.shape
    num_classes = W.shape[1]
    scores = X.dot(W)
    loss_i = np.log(np.sum(np.exp(scores), 1)) - scores[range(num_train), list(y)]
    loss += np.sum(loss_i) / num_train + 0.5 * reg * np.sum(W * W)
    derivatives = np.sum(np.exp(scores), 1).reshape(-1, 1).copy()
    derivatives = 1 / derivatives
    tmpD = derivatives.dot(np.ones((1, num_classes))) # NxC
    tmpD *= np.exp(scores)
    tmpD[range(num_train), list(y)] -= 1
    dW = X.T.dot(tmpD) / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
