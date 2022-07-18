# coding=utf-8
# Copyright 2021 The Pareto Fairness Mtl Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Head that minimizes difference of another head output on two data slices.

For example, we often want the output of another head to have similar behavior
when applied onto different slices of a data set (or the model should be fair to
difference slices). Then we could add this head to minimize this difference.
"""
import tensorflow.compat.v1 as tf

BINARY_CLASSIFICATION = 'binary_classification'
REGRESSION = 'regression'
MULTI_CLASSIFICATION = 'multi_classification'
_EPSILON = 1.0e-7


def _tile_tensors(t0, t1):
  """Tiles input tensors.

  This function tiles tensor t0 of shape [N, D] along axis = 1 and tensor t1
  of shape [N, D] along axis = 0. The intention is for applying pairwise kernel
  function k(t0[i], t0[j]), e.g. k(x, y) = x - y. Because of broadcasting,
  we can simply insert new dimensions at axis = 1 or 0.

  Args:
    t0: A tensor of shape [N, D]
    t1: A tensor of shape [N, D]

  Returns:
    A pair of tensors of shape [N, 1, D] and [1, N, D].
  """
  tiled_t0 = t0[:, tf.newaxis, :]
  tiled_t1 = t1[tf.newaxis, :, :]
  return tiled_t0, tiled_t1


def _combined_weight(w0, w1):
  """Combines input weight tensors.

  Args:
    w0: A weight tensor of shape [N, 1]
    w1: A weight tensor of shape [N, 1]

  Returns:
    A tensor w of shape [N, N] where w[i, j] = w0[i] * w0[j].
  """
  tiled_w0, tiled_w1 = _tile_tensors(w0, w1)
  return (tiled_w0 * tiled_w1)[:, :, 0]


def _matmul_n(matrices):
  """Multiplies a list of matrices."""
  result = matrices[0]
  for m in matrices[1:]:
    result = tf.matmul(result, m)
  return result


# The kernel_fn function maps tensors x of shape [N, M, D] and y of shape
# [N, M, D] to a tensor of shape [N, M].
#
# The MMD paper mentions that Gaussian and Laplace kernels are good choices to
# use.
def build_gauss_kernel(l):
  return lambda x, y: tf.exp(-tf.reduce_sum(tf.square(x - y), axis=2) / l**2)


def build_laplace_kernel(l):
  return lambda x, y: tf.exp(-tf.norm(x - y, axis=2) / l)


def build_kronecker_delta_kernel(eps=_EPSILON):
  return lambda x, y: tf.to_float(tf.norm(x - y, axis=2) <= eps)


def _preprocess_inputs(labels, features, weight_column_name):
  """Helper function to extract weights and loss_shape.

  If weights is not available, create default weights of 1.0. Also figure out
    the desired shape of loss.

  Args:
    labels: A tensor of labels.
    features: A dictionary of feature tensors.
    weight_column_name: A string of weight column name.

  Returns:
    A tuple of loss shape and weights.
  """
  if features is None or weight_column_name is None:
    weights = tf.ones_like(labels, dtype=tf.float32)[Ellipsis, 0:1]
  else:
    weights = features[weight_column_name]
  loss_shape = tf.shape(weights)
  return loss_shape, weights


def build_mmd_core_loss_fn(kernel_fn,
                           score_transform_fn=tf.sigmoid,
                           loss_name=None):
  """Builds a kernel MMD loss function.

  This builds a function that measures the Maximum Mean Discrepancy (MMD)
  between the distribution of logits on two groups of examples. The loss
  function build by this can only be used for min-diff.  For now it only
  supports hard membership labels 0 or 1. It should be easy to support soft
  membership (a probability between 0 and 1) when it is needed. See the paper
  http://papers.nips.cc/paper/3110-a-kernel-method-for-the-two-sample-problem.pdf
  for more details.

  Args:
    kernel_fn: (Function) the kernel function applied on the difference of
      logits.
    score_transform_fn: (Function) transform the logits. For binary
      classification, tests indicate that it is better to apply the MMD on the
      probability, but this might depend on the specific problem. So we allow
      the user to change this. Use None or tf.identity to use raw logits.
    loss_name: (String). Name of the min-diff head. If None, will use to 'mmd'.

  Returns:
    function that computes the loss.
  """
  loss_name = '{}_loss'.format(loss_name) if loss_name else 'mmd_loss'

  def mmd_core_loss(labels, logits, features, weight_column_name):
    """Computes loss values."""
    loss_shape, weights = _preprocess_inputs(labels, features,
                                             weight_column_name)

    if score_transform_fn is None:
      scores = logits
    else:
      scores = score_transform_fn(logits)

    weights *= tf.div_no_nan(
        tf.to_float(tf.shape(weights)[0]), tf.reduce_sum(weights))
    weights_ij = _combined_weight(weights, weights)

    pos_mask = tf.equal(tf.cast(labels, tf.float32), 1.0)  # membership 1
    pos_mask = tf.cast(pos_mask, tf.float32)
    neg_mask = tf.equal(tf.cast(labels, tf.float32), 0.0)  # membership 0
    neg_mask = tf.cast(neg_mask, tf.float32)

    pos_mean_mask = _combined_weight(pos_mask, pos_mask)
    pos_mean_weights = weights_ij * pos_mean_mask
    neg_mean_mask = _combined_weight(neg_mask, neg_mask)
    neg_mean_weights = weights_ij * neg_mean_mask
    pos_neg_mean_mask = _combined_weight(pos_mask, neg_mask)
    pos_neg_mean_weights = weights_ij * pos_neg_mean_mask

    # N*N matrix: (i,j) = kernel_fn(Xi, Xj)
    diff_kernel = kernel_fn(*_tile_tensors(scores, scores))

    pos_mean = tf.reduce_sum(pos_mean_weights * diff_kernel) / (
        tf.reduce_sum(pos_mean_weights) + 1.0e-8)
    neg_mean = tf.reduce_sum(neg_mean_weights * diff_kernel) / (
        tf.reduce_sum(neg_mean_weights) + 1.0e-8)
    pos_neg_mean = tf.reduce_sum(pos_neg_mean_weights * diff_kernel) / (
        tf.reduce_sum(pos_neg_mean_weights) + 1.0e-8)

    # MMD is actually the square root of the following quatity. However, the
    # derivative of sqrt is easy to blow up when the value is close to 0.
    # So we do not use that.
    loss = pos_mean - 2 * pos_neg_mean + neg_mean

    tf.summary.scalar(loss_name, loss)

    return loss + tf.zeros(loss_shape, dtype=tf.float32)

  return mmd_core_loss


def abs_corr_loss(labels, logits, weights, loss_name=None):
  """Computes the absolute correction loss over a mini-batch.

  This loss is meant to be used within custom tf.learn code. If you intend
  it inside a canned tf.core estimator, you should use the min diff head in
  a MultiTask estimator instead.

  Args:
    labels: The label tensor with the shape of [batch_size, 1].
    logits: The prediction tensor with the shape of [batch_size, 1].
    weights: The example weight tensor with the shape [batch_size, 1]. If None,
      will be initialized with all 1.0 weights.
    loss_name: (String). Name of the min-diff head. If None, will use to 'corr'.

  Returns:
    A single scalar loss value.
  """
  if weights is None:
    weights = tf.ones_like(labels)
  weight_sum = tf.reduce_sum(weights)

  loss_name = '{}_loss'.format(loss_name) if loss_name else 'corr_loss'

  def compute():
    """Computes the absolute value of correlation."""
    normed_weights = weights / weight_sum
    weighted_mean_labels = tf.reduce_sum(normed_weights * labels)
    weighted_mean_logits = tf.reduce_sum(normed_weights * logits)
    weighted_var_labels = tf.reduce_sum(
        normed_weights * tf.square(labels - weighted_mean_labels))
    weighted_var_logits = tf.reduce_sum(
        normed_weights * tf.square(logits - weighted_mean_logits))
    weighted_covar = tf.reduce_sum(normed_weights *
                                   (labels - weighted_mean_labels) *
                                   (logits - weighted_mean_logits))
    corr = weighted_covar / (
        tf.sqrt(weighted_var_labels + _EPSILON) *
        tf.sqrt(weighted_var_logits + _EPSILON))

    return tf.abs(corr, name=loss_name)

  loss = tf.cond(tf.greater(weight_sum, 0.), compute, lambda: 0.)
  tf.summary.scalar(loss_name, loss)
  return loss


def abs_corr_core_loss(labels, logits, features, weight_column_name):
  """Adapts the absolute correlation loss for using in the min diff head."""
  if features is None or weight_column_name is None:
    weights = tf.ones_like(labels)
  else:
    weights = features[weight_column_name]
  # Broadcast to the correct shape and return
  return abs_corr_loss(labels, logits, weights) + tf.zeros_like(logits)
