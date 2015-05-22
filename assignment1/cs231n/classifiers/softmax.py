import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  
  (C, D) = W.shape
  N = X.shape[1]

  for i in xrange(0, N):
    scores = W.dot(X[:, i])
    scores -= np.max(scores) # Correct for numerical stability

    loss -= scores[y[i]]

    sum_exp = 0.0
    for s in scores:
      sum_exp += np.exp(s)

    for j in xrange(0, C):
      dW[j, :] += 1.0 / sum_exp * np.exp(scores[j]) * X[:, i]
      if j == y[i]:
        dW[j, :] -= X[:, i]

    loss += np.log(curr_sum)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Average gradients as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization to the gradient
  dW += reg * W

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
  
  (C, D) = W.shape
  N = X.shape[1]

  scores = np.dot(W, X)
  scores -= np.max(scores) # shift by log C to avoid numerical instability

  y_mat = np.zeros(shape = (C, N))
  y_mat[y, range(N)] = 1

  # matrix of all zeros except for a single wx + log C value in each column that corresponds to the
  # quantity we need to subtract from each row of scores
  correct_wx = np.multiply(y_mat, scores)

  # create a single row of the correct wx_y + log C values for each data point
  sums_wy = np.sum(correct_wx, axis=0) # sum over each column

  exp_scores = np.exp(scores)
  sums_exp = np.sum(exp_scores, axis=0) # sum over each column
  result = np.log(sums_exp)

  result -= sums_wy

  loss = np.sum(result)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= float(N)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  sum_exp_scores = np.sum(exp_scores, axis=0) # sum over columns
  sum_exp_scores = 1.0 / (sum_exp_scores + 1e-8)

  dW = exp_scores * sum_exp_scores
  dW = np.dot(dW, X.T)
  dW -= np.dot(y_mat, X.T)

  dW /= float(N)

  # Add regularization to the gradient
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
