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

"""Util functions."""

import numpy as np
import pandas as pd
import tensorflow as tf


def get_fairness_metrics(y_pred,
                         y_true,
                         gender_numeric,
                         labeltype='classification',
                         pos_ratio=0.5,
                         thresh=None):

  """Compute fairness metrics."""
  # classification: fpr_gap, tpr_gap (need to input pos_ratio);
  # regression: mse_gap, rmse_gap
  df = pd.DataFrame({
      'y_true': y_true.numpy(),
      'y_pred': y_pred.numpy(),
      'gender_numeric': gender_numeric.numpy()
  })
  accu = 0  # Overall accuracy.
  if labeltype == 'classification':
    if thresh is None:
      # pick thresholod such that P(y=1) = P(y_hat=1)
      thresh = sorted(y_pred.numpy())[max(
          0,
          int((1 - pos_ratio) * len(y_pred.numpy())) - 1)]

    df['tp'] = 1.0 * ((df['y_true'] == 1) & (df['y_pred'] >= thresh))
    df['fp'] = 1.0 * ((df['y_true'] == 0) & (df['y_pred'] >= thresh))
    df['tn'] = 1.0 * ((df['y_true'] == 0) & (df['y_pred'] < thresh))
    df['fn'] = 1.0 * ((df['y_true'] == 1) & (df['y_pred'] < thresh))
    df_f = df[df['gender_numeric'] == 1]  # Group 1.
    df_m = df[df['gender_numeric'] == 0]  # Group 0.
    fpr_f = 1.0 * sum(df_f['fp']) / (sum(df_f['fp'] + df_f['tn']) + 1e-08)
    fpr_m = 1.0 * sum(df_m['fp']) / (sum(df_m['fp'] + df_m['tn']) + 1e-08)
    tpr_f = 1.0 * sum(df_f['tp']) / (sum(df_f['tp'] + df_f['fn']) + 1e-08)
    tpr_m = 1.0 * sum(df_m['tp']) / (sum(df_m['tp'] + df_m['fn']) + 1e-08)
    accu = 1.0 * sum(df['tp'] + df['tn']) / sum(df['tp'] + df['tn'] + df['fp'] +
                                                df['fn'])
    fpr_gap = fpr_f - fpr_m
    tpr_gap = tpr_f - tpr_m
    return accu, fpr_gap, tpr_gap
  elif labeltype == 'regression':
    df_f = df[df['gender_numeric'] == 1]
    df_m = df[df['gender_numeric'] == 0]
    mse_f = np.mean((df_f['y_true'] - df_f['y_pred'])**2)
    mse_m = np.mean((df_m['y_true'] - df_m['y_pred'])**2)
    rmse_f = np.sqrt(np.mean((df_f['y_true'] - df_f['y_pred'])**2))
    rmse_m = np.sqrt(np.mean((df_m['y_true'] - df_m['y_pred'])**2))
    mse_gap = mse_f - mse_m
    rmse_gap = rmse_f - rmse_m
    return accu, mse_gap, rmse_gap


def get_fpr_tpr(y_pred, y_true, gender_numeric, fp_func, fn_func, tp_func,
                tn_func):
  """Compute FPR and TPR."""
  # Tensor version.
  g1_ind = tf.where(tf.equal(gender_numeric, 1))[:, -1]
  g0_ind = tf.where(tf.equal(gender_numeric, 0))[:, -1]

  fp_func.reset_states()
  fp_func.update_state(tf.gather(y_true, g1_ind), tf.gather(y_pred, g1_ind))
  fp_g1 = fp_func.result()
  fp_func.reset_states()
  fp_func.update_state(tf.gather(y_true, g0_ind), tf.gather(y_pred, g0_ind))
  fp_g0 = fp_func.result()

  fn_func.reset_states()
  fn_func.update_state(tf.gather(y_true, g1_ind), tf.gather(y_pred, g1_ind))
  fn_g1 = fn_func.result()
  fn_func.reset_states()
  fn_func.update_state(tf.gather(y_true, g0_ind), tf.gather(y_pred, g0_ind))
  fn_g0 = fn_func.result()

  tp_func.reset_states()
  tp_func.update_state(tf.gather(y_true, g1_ind), tf.gather(y_pred, g1_ind))
  tp_g1 = tp_func.result()
  tp_func.reset_states()
  tp_func.update_state(tf.gather(y_true, g0_ind), tf.gather(y_pred, g0_ind))
  tp_g0 = tp_func.result()

  tn_func.reset_states()
  tn_func.update_state(tf.gather(y_true, g1_ind), tf.gather(y_pred, g1_ind))
  tn_g1 = tn_func.result()
  tn_func.reset_states()
  tn_func.update_state(tf.gather(y_true, g0_ind), tf.gather(y_pred, g0_ind))
  tn_g0 = tn_func.result()

  fpr_g1 = fp_g1 / (fp_g1 + tn_g1 + 1e-08)
  fpr_g0 = fp_g0 / (fp_g0 + tn_g0 + 1e-08)
  tpr_g1 = tp_g1 / (tp_g1 + fn_g1 + 1e-08)
  tpr_g0 = tp_g0 / (tp_g0 + fn_g0 + 1e-08)

  fpr_gap = fpr_g1 - fpr_g0
  tpr_gap = tpr_g1 - tpr_g0

  return fpr_gap, tpr_gap


def corr(x, y):
  numerator = tf.reduce_mean(tf.multiply(
      x, y)) - tf.reduce_mean(x) * tf.reduce_mean(y)
  denominator = tf.math.reduce_std(x) * tf.math.reduce_std(y)
  return numerator / (denominator + 1e-06)
