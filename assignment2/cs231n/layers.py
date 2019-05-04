from builtins import range
import numpy as np

# def softmax_loss(scores, y):
#     loss = 0.0
#
#     #############################################################################
#     # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
#     # Store the loss in loss and the gradient in dW. If you are not careful     #
#     # here, it is easy to run into numeric instability. Don't forget the        #
#     # regularization!                                                           #
#     #############################################################################
#     shift_scores = scores - np.max(scores, axis=1, keepdims=True)
#
#     num_train, num_classes = shift_scores.shape
#     loss_i = np.log(np.sum(np.exp(shift_scores), axis=1)) - shift_scores[range(num_train), list(y)]
#     loss += np.sum(loss_i) / num_train
#     derivatives = 1 / np.sum(np.exp(shift_scores), axis=1, keepdims=True)
#     dscores = np.exp(shift_scores) / derivatives
#     dscores[range(num_train), list(y)] -= 1
#     #############################################################################
#     #                          END OF YOUR CODE                                 #
#     #############################################################################
#     return loss, dscores

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    D, M = w.shape
    tmp_x = x.reshape(N, D)
    out = tmp_x.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    D, M = w.shape
    tmp_x = x.reshape(N, D)
    dw = tmp_x.T.dot(dout)
    db = np.sum(dout, axis=0).T
    tmp_dx = dout.dot(w.T)
    dx = tmp_dx.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = (x > 0) * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        running_mean = momentum * running_mean + (1 - momentum) * np.mean(x, axis=0)
        running_var = momentum * running_var + (1 - momentum) * np.var(x, axis=0)

        Ex = np.ones((N, 1)).dot((np.mean(x, axis=0)).reshape((1, D)))
        root_Varx = np.ones((N, 1)).dot(np.sqrt(np.var(x, axis=0) + eps).reshape((1, D)))
        x_ = (x - Ex) / root_Varx
        out = x_ * (np.ones((N, 1)).dot(gamma.reshape(1, D))) + (np.ones((N, 1)).dot(beta.reshape(1, D)))
        cache['eps'] = eps
        cache['x'] = x
        cache['x_'] = x_
        cache['gamma'] = gamma
        cache['var'] = np.var(x, axis=0)
        cache['mean'] = np.mean(x, axis=0)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        Ex = np.ones((N, 1)).dot(running_mean.reshape((1, D)))
        root_Varx = np.ones((N, 1)).dot(np.sqrt(running_var + eps).reshape((1, D)))
        out = ((x - Ex) / root_Varx) * (np.ones((N, 1)).dot(gamma.reshape(1, D))) + (np.ones((N, 1)).dot(beta.reshape(1, D)))
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N, D = dout.shape
    gamma = cache['gamma']
    x = cache['x']
    x_ = cache['x_']
    eps = cache['eps']
    var = cache['var'].reshape(1, D)
    Ex = np.ones((N, 1)).dot(cache['mean'].reshape(1, D)) # NxD
    sigma_square = np.ones((N, 1)).dot(var) # NxD

    dgamma = np.sum(x_ * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_ = np.ones((N, 1)).dot(gamma.reshape(1, D)) * dout # NxD
    dvar = np.sum(dx_*(x-Ex)*(-0.5)*((sigma_square + eps)**(-1.5)), axis=0).reshape(1, D) # 1xD
    dmean = (np.sum(dx_*(-1)/np.sqrt(sigma_square + eps), axis=0)).reshape(1, D) - 2 * dvar * np.sum(x-Ex, axis=0).reshape(1, D) / N # 1xD
    dx = dx_ / np.sqrt(sigma_square + eps) + (np.ones((N, 1)).dot(dvar) * 2 * (x-Ex) + np.ones((N, 1)).dot(dmean)) / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    return batchnorm_backward(dout, cache)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, {}
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    N, D = x.shape
    Ex = (np.mean(x, axis=1)).reshape(N, 1).dot(np.ones((1, D)))
    root_Varx = (np.sqrt(np.var(x, axis=1) + eps)).reshape(N, 1).dot(np.ones((1, D)))
    x_ = (x - Ex) / root_Varx
    out = x_ * (np.ones((N, 1)).dot(gamma.reshape(1, D))) + (np.ones((N, 1)).dot(beta.reshape(1, D)))
    cache['eps'] = eps
    cache['x'] = x
    cache['x_'] = x_
    cache['gamma'] = gamma
    cache['var'] = np.var(x, axis=1)
    cache['mean'] = np.mean(x, axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    N, D = dout.shape
    gamma = cache['gamma']
    x = cache['x']
    x_ = cache['x_']
    eps = cache['eps']
    var = cache['var'].reshape(N, 1)
    Ex = (cache['mean'].reshape(N, 1)).dot(np.ones((1, D))) # NxD
    sigma_square = var.dot(np.ones((1, D))) # NxD

    dgamma = np.sum(x_ * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_ = np.ones((N, 1)).dot(gamma.reshape(1, D)) * dout # NxD
    dvar = np.sum(dx_*(x-Ex)*(-0.5)*((sigma_square + eps)**(-1.5)), axis=1).reshape(N, 1) # Nx1
    dmean = (np.sum(dx_*(-1)/np.sqrt(sigma_square + eps), axis=1)).reshape(N, 1) - 2 * dvar * np.sum(x-Ex, axis=1).reshape(N, 1) / N # Nx1
    dx = dx_ / np.sqrt(sigma_square + eps) + (dvar.dot(np.ones((1, D))) * 2 * (x-Ex) + (dmean.dot(np.ones((1, D))))) / D
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C_, HH, WW = w.shape
    # refer to the annotation
    if C_ != C:
        return None, None

    stride = conv_param['stride']
    pad = conv_param['pad']

    x_with_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0.0, 0.0))
    H_ = int(1 + (H + 2 * pad - HH) / stride)
    W_ = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_, W_))
    for row in range(H_):
        for col in range(W_):
            H_begin = row * stride
            W_begin = col * stride
            H_end = H_begin + HH
            W_end = W_begin + WW
            # for case_idx in range(N):
                # active_grids = np.tile(x_with_pad[case_idx, :, H_begin:H_end, W_begin:W_end].reshape(C, HH, WW), (F, 1, 1, 1))
                # out[case_idx, :, row, col] = np.sum(active_grids * w, axis=(1, 2, 3)) + b.reshape(1, F)
            active_pixels = np.tile(x_with_pad[:, :, H_begin:H_end, W_begin:W_end].reshape(N, 1, C, HH, WW), (1, F, 1, 1, 1))
            out[:, :, row, col] = np.tile(b, (N, 1)) + np.sum(active_pixels * np.tile(w.reshape(1, F, C, HH, WW), (N, 1, 1, 1, 1)), axis=(2, 3, 4))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def get_Xij_is_active_to_filter(i, j, H, W, HH, WW, pad, stride, H_, W_):
    """
    return the active indexes in filter
    """
    _i, _j = i + pad, j + pad
    # _i-HH <= n*stride <= _i
    H_ns = range(max(int((_i-HH+stride)/stride), 0), min(int(_i/stride)+1, H_))
    W_ns = range(max(int((_j-WW+stride)/stride), 0), min(int(_j/stride)+1, W_))
    indexes = []
    for k in H_ns:
        for t in W_ns:
            indexes.append(((k, t), (_i-k*stride, _j-t*stride)))
    return indexes


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape # FxCxHHxWW
    _, _, H_, W_ = dout.shape # NxFxH'xW'

    stride = conv_param['stride']
    pad = conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))
    # dw[f,c,hh,ww] is the sum of elements of x which is active for w[f,c,hh,ww]
    # dx[n,c,h,w] is the sum of elements of w which counted with dx[n,c,h,w]
    # note, the numbers of counts for every element may be different, because of the stride
    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    for i in range(H):
        for j in range(W):
            indexes_in_filter = get_Xij_is_active_to_filter(i, j, H, W, HH, WW, pad, stride, H_, W_)
            for idx in indexes_in_filter:
                dx[:, :, i, j] += dout[:, :, idx[0][0], idx[0][1]].dot(w[:, :, idx[1][0], idx[1][1]])
                dw[:, :, idx[1][0], idx[1][1]] += dout[:, :, idx[0][0], idx[0][1]].T.dot(x[:, :, i, j])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']

    H_ = 1 + int((H - ph) / stride)
    W_ = 1 + int((W - pw) / stride)
    out = np.zeros((N, C, H_, W_))
    for row in range(H_):
        for col in range(W_):
            H_begin = row * stride
            W_begin = col * stride
            H_end = H_begin + ph
            W_end = W_begin + pw
            out[:, :, row, col] = np.max(x[:, :, H_begin:H_end, W_begin:W_end], axis=(2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']
    _, _, H_, W_ = dout.shape

    dx = np.zeros(x.shape)
    for row in range(H_):
        for col in range(W_):
            H_begin = row * stride
            W_begin = col * stride
            H_end = H_begin + ph
            W_end = W_begin + pw

            # max_indexes = [[(0, 0) for k in range(C)] for t in range(N)]
            # # find the max idx for every NxC data
            # for n in range(N):
            #     for c in range(C):
            #         max_indexes[n][c] = (H_begin, W_begin)
            #         for h_idx in range(H_begin, H_end):
            #             for w_idx in range(W_begin, W_end):
            #                 if x[n, c, h_idx, w_idx] > x[n, c, max_indexes[n][c][0], max_indexes[n][c][1]]:
            #                     max_indexes[n][c] = (h_idx, w_idx)
            # for n in range(N):
            #     for c in range(C):
            #         dx[n, c, max_indexes[n][c][0], max_indexes[n][c][1]] += dout[n, c, row, col]

            active_x = x[:, :, H_begin:H_end, W_begin:W_end]
            max_map = (active_x == np.max(active_x, axis=(2, 3), keepdims=True))

            for n in range(N):
                for c in range(C):
                    dx[n, c, H_begin:H_end, W_begin:W_end] += dout[n, c, row, col] * max_map[n, c, :, :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    tmp_x = x.reshape(N, C*H*W)
    tmp_gamma = (np.tile(gamma.reshape(C, 1), (1, H*W))).reshape(C*H*W)
    tmp_beta = (np.tile(beta.reshape(C, 1), (1, H*W))).reshape(C*H*W)
    tmp_out, cache = batchnorm_forward(tmp_x, tmp_gamma, tmp_beta, bn_param)
    out = tmp_out.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    tmp_dout = dout.reshape(N, C*H*W)
    tmp_dx, tmp_dgamma, tmp_dbeta = batchnorm_backward(tmp_dout, cache)
    dx = tmp_dx.reshape(N, C, H, W)
    dgamma = (tmp_dgamma.reshape(C, H*W).T)[0]
    dbeta = (tmp_dbeta.reshape(C, H*W).T)[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
