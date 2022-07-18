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

"""Experiments on UCI-Adult Dataset."""

import collections

from absl import app
import min_diff_losses
import pandas as pd
import sklearn
import tensorflow as tf
import utils


TRAIN_DATASET_SIZE = 800
EVAL_DATASET_SIZE = 200
EMBEDDING_SIZE = 1
NUM_FEATURE_COLS = [
    'duration_in_month', 'installment_rate', 'resident_since', 'age',
    'num_credits', 'num_liable'
]
CAT_FEATURE_COLS = [
    'status_checking_account', 'gender', 'credit_history', 'purpose', 'savings',
    'employement_since', 'debters', 'property', 'other_installments', 'housing',
    'job', 'telephone', 'foreign_worker'
]
# Used for creating one-hot encoding for categorical features:
VOCAB_SIZE_DICT = {
    'status_checking_account': 4,
    'duration_in_month': 33,
    'credit_history': 5,
    'purpose': 10,
    'credit_amount': 921,
    'savings': 5,
    'employement_since': 5,
    'installment_rate': 4,
    'gender': 2,
    'debters': 3,
    'resident_since': 4,
    'property': 4,
    'age': 53,
    'other_installments': 3,
    'housing': 3,
    'num_credits': 4,
    'job': 4,
    'num_liable': 2,
    'telephone': 2,
    'foreign_worker': 2
}


def load_dataset(batch_size):
  """Load dataset."""
  data_df = pd.read_csv(
      open('.../german.data'), header=None,
      delimiter=' ')  # Provide path to dataset here.
  data_df.rename(
      columns={
          0: 'status_checking_account',
          1: 'duration_in_month',
          2: 'credit_history',
          3: 'purpose',
          4: 'credit_amount',
          5: 'savings',
          6: 'employement_since',
          7: 'installment_rate',
          8: 'gender',
          9: 'debters',
          10: 'resident_since',
          11: 'property',
          12: 'age',
          13: 'other_installments',
          14: 'housing',
          15: 'num_credits',
          16: 'job',
          17: 'num_liable',
          18: 'telephone',
          19: 'foreign_worker',
          20: 'is_good_loan'
      },
      inplace=True)
  data_df.replace(
      {
          'gender': {
              'A91': 'male',
              'A92': 'female',
              'A93': 'male',
              'A94': 'male',
              'A95': 'female'
          }
      },
      inplace=True)
  data_df['is_good_loan'] = data_df[
      'is_good_loan'] - 1  # Make it a binary response.
  data_df['is_high_credit'] = 1 * (data_df['credit_amount'] > 2000)

  # Normalize continuous features.
  for col in NUM_FEATURE_COLS:
    data_df[col] = (data_df[col] - data_df[col].mean()) / data_df[col].std()

  for col in CAT_FEATURE_COLS:  # convert categorical features to int
    data_df[col] = pd.Categorical(
        data_df[col], categories=data_df[col].unique()).codes

  df_tr, df_tst = sklearn.model_selection.train_test_split(
      data_df, test_size=0.2)
  Dataset = collections.namedtuple('Dataset', ['train', 'valid', 'test'])
  return Dataset(
      tf.data.Dataset.from_tensor_slices(
          dict(df_tr)).shuffle(TRAIN_DATASET_SIZE).batch(batch_size),
      tf.data.Dataset.from_tensor_slices(
          dict(df_tst)).shuffle(EVAL_DATASET_SIZE).batch(batch_size),
      tf.data.Dataset.from_tensor_slices(
          dict(df_tst)).shuffle(EVAL_DATASET_SIZE).batch(batch_size))


def process_input(f):
  """Process input.

  Args:
    f: dataframe of input.

  Returns:
    feature dict and label dict.

  """
  features = {
      'status_checking_account': f['status_checking_account'],
      'duration_in_month': f['duration_in_month'],
      'credit_history': f['credit_history'],
      'purpose': f['purpose'],
      'savings': f['savings'],
      'employement_since': f['employement_since'],
      'installment_rate': f['installment_rate'],
      'gender': f['gender'],
      'debters': f['debters'],
      'resident_since': f['resident_since'],
      'property': f['property'],
      'age': f['age'],
      'other_installments': f['other_installments'],
      'housing': f['housing'],
      'num_credits': f['num_credits'],
      'job': f['job'],
      'num_liable': f['num_liable'],
      'telephone': f['telephone'],
      'foreign_worker': f['foreign_worker']
  }

  labels = {
      'is_good_loan': f['is_good_loan'],
      'is_high_credit': f['is_high_credit']
  }
  return features, labels


def get_feature_columns():
  return [
      # Numerical features.
      tf.feature_column.numeric_column('duration_in_month'),
      tf.feature_column.numeric_column('installment_rate'),
      tf.feature_column.numeric_column('resident_since'),
      tf.feature_column.numeric_column('age'),
      tf.feature_column.numeric_column('num_credits'),
      tf.feature_column.numeric_column('num_liable'),

      # Categorical features.
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'gender', VOCAB_SIZE_DICT['gender']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'status_checking_account',
              VOCAB_SIZE_DICT['status_checking_account']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'credit_history', VOCAB_SIZE_DICT['credit_history']),
          EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'purpose', VOCAB_SIZE_DICT['purpose']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'savings', VOCAB_SIZE_DICT['savings']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'employement_since', VOCAB_SIZE_DICT['employement_since']),
          EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'debters', VOCAB_SIZE_DICT['debters']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'property', VOCAB_SIZE_DICT['property']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'other_installments', VOCAB_SIZE_DICT['other_installments']),
          EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'housing', VOCAB_SIZE_DICT['housing']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'job', VOCAB_SIZE_DICT['job']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'telephone', VOCAB_SIZE_DICT['telephone']), EMBEDDING_SIZE),
      tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_identity(
              'foreign_worker', VOCAB_SIZE_DICT['foreign_worker']),
          EMBEDDING_SIZE)
  ]


def get_positive_ratio(dataset, binary_label):
  """Get the positive ratio from the training data set."""
  label_tot = []
  for f in dataset.train:
    _, labels = process_input(f)
    label_tot = tf.concat(
        [label_tot, tf.reshape(labels[binary_label], [-1])], 0)
  pos_ratio = tf.cast(tf.reduce_sum(label_tot), tf.float32) / tf.cast(
      tf.size(label_tot), tf.float32)
  return pos_ratio


@tf.function()
def get_gender_numeric(x):
  if tf.equal(x, 1):
    return 1.0
  else:
    return 0.0


def train(params):
  """Main training function."""

  tf.logging.info(params)

  # Define tasks for MTL.
  t1 = params.t1
  t2 = params.t2

  # Get dataset.
  dataset_fn = load_dataset
  dataset = dataset_fn(params.batch_size)

  # Compute positive label ratios, used to determine classification threshold.
  pos_t1_ratio = get_positive_ratio(dataset, t1)
  pos_t2_ratio = get_positive_ratio(dataset, t2)

  # Define confusion metrics.
  fp_func_t1 = tf.keras.metrics.FalsePositives(
      thresholds=pos_t1_ratio.numpy(), dtype=tf.float32)
  fn_func_t1 = tf.keras.metrics.FalseNegatives(
      thresholds=pos_t1_ratio.numpy(), dtype=tf.float32)
  tp_func_t1 = tf.keras.metrics.TruePositives(
      thresholds=pos_t1_ratio.numpy(), dtype=tf.float32)
  tn_func_t1 = tf.keras.metrics.TrueNegatives(
      thresholds=pos_t1_ratio.numpy(), dtype=tf.float32)

  fp_func_t2 = tf.keras.metrics.FalsePositives(
      thresholds=pos_t2_ratio.numpy(), dtype=tf.float32)
  fn_func_t2 = tf.keras.metrics.FalseNegatives(
      thresholds=pos_t2_ratio.numpy(), dtype=tf.float32)
  tp_func_t2 = tf.keras.metrics.TruePositives(
      thresholds=pos_t2_ratio.numpy(), dtype=tf.float32)
  tn_func_t2 = tf.keras.metrics.TrueNegatives(
      thresholds=pos_t2_ratio.numpy(), dtype=tf.float32)

  shared_units = [int(i) for i in params.shared_units_str.split('_') if i]
  t1_units = [int(i) for i in params.t1_units_str.split('_') if i]
  t2_units = [int(i) for i in params.t2_units_str.split('_') if i]

  input_layer = tf.keras.Sequential(
      [tf.keras.layers.DenseFeatures(get_feature_columns())])
  shared_bottom_layer = tf.keras.Sequential(
      [tf.keras.layers.Dense(u, tf.nn.relu) for u in shared_units])
  t1_tower_layer = tf.keras.Sequential(
      [tf.keras.layers.Dense(u, tf.nn.relu) for u in t1_units])
  t1_tower_final = tf.keras.Sequential([tf.keras.layers.Dense(1)])
  t2_tower_layer = tf.keras.Sequential(
      [tf.keras.layers.Dense(u, tf.nn.relu) for u in t2_units])
  t2_tower_final = tf.keras.Sequential([tf.keras.layers.Dense(1)])

  optimizer = tf.train.AdagradOptimizer(params.lr)

  def model(features, labels):
    input_layer_output = input_layer(features)
    shared_bottom_layer_output = shared_bottom_layer(input_layer_output)
    t1_tower_layer_output = t1_tower_layer(shared_bottom_layer_output)
    t1_logits = t1_tower_final(t1_tower_layer_output)
    t2_tower_layer_output = t2_tower_layer(shared_bottom_layer_output)
    t2_logits = t2_tower_final(t2_tower_layer_output)

    if params.t1_type == 'classification':
      t1_loss = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.reshape(tf.cast(labels[t1], tf.float32), (-1, 1)),
              logits=t1_logits))
      t1_accu = tf.reduce_sum(
          tf.cast(
              tf.cast(tf.math.greater(t1_logits, 0.0),
                      tf.float32) == tf.reshape(
                          tf.cast(labels[t1], tf.float32), (-1, 1)),
              tf.float32))  # Note: this accuracy is based on threshold=0
      t1_pred = tf.math.sigmoid(t1_logits)
    elif params.t1_type == 'regression':
      t1_loss = tf.reduce_sum(
          tf.math.square(
              tf.reshape(tf.cast(labels[t1], tf.float32), (-1, 1)) - t1_logits))
      t1_accu = tf.zeros_like(t1_loss)
      t1_pred = t1_logits

    if params.t2_type == 'classification':
      t2_loss = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.reshape(tf.cast(labels[t2], tf.float32), (-1, 1)),
              logits=t2_logits))
      t2_accu = tf.reduce_sum(
          tf.cast(
              tf.cast(tf.math.greater(t2_logits, 0.0),
                      tf.float32) == tf.reshape(
                          tf.cast(labels[t2], tf.float32), (-1, 1)),
              tf.float32))
      t2_pred = tf.math.sigmoid(t2_logits)
    elif params.t2_type == 'regression':
      t2_loss = tf.reduce_sum(
          tf.math.square(
              tf.reshape(tf.cast(labels[t2], tf.float32), (-1, 1)) - t2_logits))
      t2_accu = tf.zeros_like(t2_loss)
      t2_pred = t2_logits

    # Compute fairness loss
    gender_numeric_batch = tf.map_fn(
        fn=get_gender_numeric,
        elems=features['gender'],
        fn_output_signature=tf.float32)
    gender_numeric = tf.reshape(gender_numeric_batch, [-1, 1])

    kernel_fn = min_diff_losses.build_gauss_kernel(
        params.mmd_kernel_decay_length)
    mmd_loss_fn = min_diff_losses.build_mmd_core_loss_fn(
        kernel_fn, score_transform_fn=None)

    t1_neg_ind = tf.where(tf.equal(labels[t1], 0))[:, -1]
    t2_neg_ind = tf.where(tf.equal(labels[t2], 0))[:, -1]
    # For MTA-F.
    only_t1_neg_ind = tf.where(
        tf.logical_and(tf.equal(labels[t2], 1), tf.equal(labels[t1], 0)))[:, -1]
    only_t2_neg_ind = tf.where(
        tf.logical_and(tf.equal(labels[t2], 0), tf.equal(labels[t1], 1)))[:, -1]

    sb_mindiff = 0

    # Different fairness losses.
    if 'mmd_baseline' in params.mindiff_type:
      t1_mindiff = tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, t1_neg_ind),
              logits=tf.gather(t1_pred, t1_neg_ind),
              features=None,
              weight_column_name=None))
      t2_mindiff = tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, t2_neg_ind),
              logits=tf.gather(t2_pred, t2_neg_ind),
              features=None,
              weight_column_name=None))
    elif 'mmd_mtaf' in params.mindiff_type:
      t1_mindiff_full = tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, t1_neg_ind),
              logits=tf.gather(t1_pred, t1_neg_ind),
              features=None,
              weight_column_name=None))
      t2_mindiff_full = tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, t2_neg_ind),
              logits=tf.gather(t2_pred, t2_neg_ind),
              features=None,
              weight_column_name=None))
      # Exclusive
      t1_mindiff = params.lam_headratio_t1 * tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, only_t1_neg_ind),
              logits=tf.gather(t1_pred, only_t1_neg_ind),
              features=None,
              weight_column_name=None))
      t2_mindiff = params.lam_headratio_t2 * tf.reduce_sum(
          mmd_loss_fn(
              labels=tf.gather(gender_numeric, only_t2_neg_ind),
              logits=tf.gather(t2_pred, only_t2_neg_ind),
              features=None,
              weight_column_name=None))
      sb_mindiff = (
          params.lam_t1 * params.alpha * (t1_mindiff_full - t1_mindiff) +
          params.lam_t2 * (1.0 - params.alpha) * (t2_mindiff_full - t2_mindiff))
    elif 'corr_baseline' in params.mindiff_type:
      t1_mindiff = tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t1_pred, t1_neg_ind),
                  tf.gather(gender_numeric, t1_neg_ind))) +
          tf.zeros(tf.shape(t1_neg_ind), dtype=tf.float32))
      t2_mindiff = tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t2_pred, t2_neg_ind),
                  tf.gather(gender_numeric, t2_neg_ind))) +
          tf.zeros(tf.shape(t2_neg_ind), dtype=tf.float32))
    elif 'corr_mtaf' in params.mindiff_type:
      t1_mindiff_full = tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t1_pred, t1_neg_ind),
                  tf.gather(gender_numeric, t1_neg_ind))) +
          tf.zeros(tf.shape(t1_neg_ind), dtype=tf.float32))
      t2_mindiff_full = tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t2_pred, t2_neg_ind),
                  tf.gather(gender_numeric, t2_neg_ind))) +
          tf.zeros(tf.shape(t2_neg_ind), dtype=tf.float32))
      # Exclusive
      t1_mindiff = params.lam_headratio_t1 * tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t1_pred, only_t1_neg_ind),
                  tf.gather(gender_numeric, only_t1_neg_ind))) +
          tf.zeros(tf.shape(only_t1_neg_ind), dtype=tf.float32))
      t2_mindiff = params.lam_headratio_t2 * tf.reduce_sum(
          tf.abs(
              utils.corr(
                  tf.gather(t2_pred, only_t2_neg_ind),
                  tf.gather(gender_numeric, only_t2_neg_ind))) +
          tf.zeros(tf.shape(only_t2_neg_ind), dtype=tf.float32))
      sb_mindiff = (
          params.lam_t1 * params.alpha * (t1_mindiff_full - t1_mindiff) +
          params.lam_t2 * (1.0 - params.alpha) * (t2_mindiff_full - t2_mindiff))
    elif 'fprgap_baseline' in params.mindiff_type:
      t1_mindiff = tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(t1_pred, labels[t1], gender_numeric_batch,
                                fp_func_t1, fn_func_t1, tp_func_t1, tn_func_t1)
              [0]) + tf.zeros(tf.shape(t1_pred), dtype=tf.float32))
      t2_mindiff = tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(t2_pred, labels[t2], gender_numeric_batch,
                                fp_func_t2, fn_func_t2, tp_func_t2, tn_func_t2)
              [0]) + tf.zeros(tf.shape(t2_pred), dtype=tf.float32))
    elif 'fprgap_mtaf' in params.mindiff_type:
      t1_mindiff_full = tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(t1_pred, labels[t1], gender_numeric_batch,
                                fp_func_t1, fn_func_t1, tp_func_t1, tn_func_t1)
              [0]) + tf.zeros(tf.shape(t1_pred), dtype=tf.float32))
      t2_mindiff_full = tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(t2_pred, labels[t2], gender_numeric_batch,
                                fp_func_t2, fn_func_t2, tp_func_t2, tn_func_t2)
              [0]) + tf.zeros(tf.shape(t2_pred), dtype=tf.float32))
      # Exclusive
      t1_mindiff = params.lam_headratio_t1 * tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(
                  tf.gather(t1_pred, only_t1_neg_ind),
                  tf.gather(labels[t1], only_t1_neg_ind),
                  tf.gather(gender_numeric_batch, only_t1_neg_ind), fp_func_t1,
                  fn_func_t1, tp_func_t1, tn_func_t1)[0]) +
          tf.zeros(tf.shape(only_t1_neg_ind), dtype=tf.float32))
      t2_mindiff = params.lam_headratio_t2 * tf.reduce_sum(
          tf.abs(
              utils.get_fpr_tpr(
                  tf.gather(t2_pred, only_t2_neg_ind),
                  tf.gather(labels[t2], only_t2_neg_ind),
                  tf.gather(gender_numeric_batch, only_t2_neg_ind), fp_func_t2,
                  fn_func_t2, tp_func_t2, tn_func_t2)[0]) +
          tf.zeros(tf.shape(only_t2_neg_ind), dtype=tf.float32))
      sb_mindiff = (
          params.lam_t1 * params.alpha * (t1_mindiff_full - t1_mindiff) +
          params.lam_t2 * (1.0 - params.alpha) * (t2_mindiff_full - t2_mindiff))
    else:
      raise ValueError('Invalid fairness loss type')

    if 'mtaf' in params.mindiff_type:
      t1_loss_final = params.alpha * (t1_loss + params.lam_t1 * t1_mindiff)
      t2_loss_final = (1 - params.alpha) * (
          t2_loss + params.lam_t2 * t2_mindiff)
      sb_loss_final = params.alpha * t1_loss + (
          1 - params.alpha) * t2_loss + sb_mindiff
    else:
      t1_loss_final = params.alpha * (t1_loss + params.lam_t1 * t1_mindiff)
      t2_loss_final = (1 - params.alpha) * (
          t2_loss + params.lam_t2 * t2_mindiff)
      sb_loss_final = params.alpha * (t1_loss + params.lam_t1 * t1_mindiff) + (
          1 - params.alpha) * (
              t2_loss + params.lam_t2 * t2_mindiff)

    loss = params.alpha * t1_loss_final + (1 - params.alpha) * t2_loss_final

    res = {
        't1_loss_final': t1_loss_final,
        't2_loss_final': t2_loss_final,
        'sb_loss_final': sb_loss_final,
        'loss': loss,
        't1_loss': t1_loss,
        't2_loss': t2_loss,
        't1_accu': t1_accu,
        't2_accu': t2_accu,
        't1_mindiff': t1_mindiff,
        't2_mindiff': t2_mindiff,
        't1_pred': t1_pred,
        't2_pred': t2_pred
    }
    return res

  @tf.function()
  def plain_train_step(features, labels):
    with tf.GradientTape(persistent=True) as tape:
      res = model(features, labels)

      sb_variables = (
          input_layer.trainable_weights +  # Shared input.
          shared_bottom_layer.trainable_weights)  # Shared bottom output.
      t1_variables = (
          t1_tower_layer.trainable_weights
          +  # Task 1 tower final layer before output.
          t1_tower_final.trainable_weights)  # Task 1 tower final output.
      t2_variables = (
          t2_tower_layer.trainable_weights
          +  # Task 2 tower final layer before output.
          t2_tower_final.trainable_weights)  # Task 2 tower final output.
      gradients_sb = tape.gradient(res['sb_loss_final'], sb_variables)
      gradients_t1 = tape.gradient(res['t1_loss_final'], t1_variables)
      gradients_t2 = tape.gradient(res['t2_loss_final'], t2_variables)
      optimizer.apply_gradients(zip(gradients_sb, sb_variables))
      optimizer.apply_gradients(zip(gradients_t1, t1_variables))
      optimizer.apply_gradients(zip(gradients_t2, t2_variables))

    return res

  @tf.function()
  def eval_step(features, labels):
    res = model(features, labels)
    return res

  # Track total results.
  total_train = {
      't1_loss': [],
      't2_loss': [],
      't1_mindiff': [],
      't2_mindiff': []
  }
  total_eval = {
      't1_loss': [],
      't2_loss': [],
      't1_accu': [],
      't2_accu': [],
      't1_mindiff': [],
      't2_mindiff': [],
      't1_fairness_gap': [],
      't2_fairness_gap': []
  }

  for current_epoch in range(params.num_epochs):
    print('epoch: {}'.format(current_epoch))
    epoch_loss, epoch_t1_loss, epoch_t2_loss = 0, 0, 0
    epoch_t1_mindiff, epoch_t2_mindiff = 0, 0
    for f in dataset.train:
      features, labels = process_input(f)
      res = plain_train_step(features, labels)

      # Only eval once
      if current_epoch < params.num_epochs - 1:
        continue

      epoch_loss += res['loss'] / TRAIN_DATASET_SIZE
      epoch_t1_loss += res['t1_loss'] / TRAIN_DATASET_SIZE
      epoch_t2_loss += res['t2_loss'] / TRAIN_DATASET_SIZE
      epoch_t1_mindiff += res['t1_mindiff'] / TRAIN_DATASET_SIZE
      epoch_t2_mindiff += res['t2_mindiff'] / TRAIN_DATASET_SIZE

    # Only eval once
    if current_epoch < params.num_epochs - 1:
      continue

    tf.logging.info(
        'total train loss: %s || Task 1 loss: %s || Task 2 loss: %s',
        epoch_loss, epoch_t1_loss, epoch_t2_loss)
    tf.logging.info(
        'train Task 1 mindiff loss: %s || train Task 2 mindiff loss: %s',
        epoch_t1_mindiff, epoch_t2_mindiff)

    # Track total metrics.
    total_train['t1_loss'].append(epoch_t1_loss.numpy())
    total_train['t2_loss'].append(epoch_t2_loss.numpy())
    total_train['t1_mindiff'].append(epoch_t1_mindiff.numpy())
    total_train['t2_mindiff'].append(epoch_t2_mindiff.numpy())

    # Eval.
    epoch_eval = {
        'loss': 0,
        't1_loss': 0,
        't2_loss': 0,
        't1_accu': 0.0,
        't2_accu': 0.0,
        't1_mindiff': 0,
        't2_mindiff': 0
    }

    # aggregate prediction and label here
    t1_pred_tot = []
    t2_pred_tot = []
    t1_label_tot = []
    t2_label_tot = []
    gender_numeric_tot = []

    for f in dataset.valid if params.use_valid else dataset.test:
      features, labels = process_input(f)
      res = eval_step(features, labels)

      epoch_eval['loss'] += res['loss'] / EVAL_DATASET_SIZE
      epoch_eval['t1_loss'] += res['t1_loss'] / EVAL_DATASET_SIZE
      epoch_eval['t2_loss'] += res['t2_loss'] / EVAL_DATASET_SIZE
      epoch_eval['t1_mindiff'] += res['t1_mindiff'] / EVAL_DATASET_SIZE
      epoch_eval['t2_mindiff'] += res['t2_mindiff'] / EVAL_DATASET_SIZE

      # Compute fairness metrics: first aggregate all labels and predictions
      t1_pred_tot = tf.concat(
          [t1_pred_tot, tf.reshape(res['t1_pred'], [-1])], 0)
      t2_pred_tot = tf.concat(
          [t2_pred_tot, tf.reshape(res['t2_pred'], [-1])], 0)
      t1_label_tot = tf.concat(
          [t1_label_tot,
           tf.reshape(tf.cast(labels[t1], tf.float32), [-1])], 0)
      t2_label_tot = tf.concat(
          [t2_label_tot,
           tf.reshape(tf.cast(labels[t2], tf.float32), [-1])], 0)
      gender_numeric_tot = tf.concat([
          gender_numeric_tot,
          tf.map_fn(
              fn=get_gender_numeric,
              elems=features['gender'],
              fn_output_signature=tf.float32)
      ], 0)

    t1_accu, t1_fairness_gap, _ = utils.get_fairness_metrics(
        t1_pred_tot,
        t1_label_tot,
        gender_numeric_tot,
        params.t1_type,
        pos_ratio=pos_t1_ratio)
    t2_accu, t2_fairness_gap, _ = utils.get_fairness_metrics(
        t2_pred_tot,
        t2_label_tot,
        gender_numeric_tot,
        params.t2_type,
        pos_ratio=pos_t2_ratio)
    epoch_eval['t1_fairness_gap'] = t1_fairness_gap
    epoch_eval['t2_fairness_gap'] = t2_fairness_gap
    epoch_eval['t1_accu'] = t1_accu
    epoch_eval['t2_accu'] = t2_accu

    tf.logging.info('eval loss: %s || Task 1 loss: %s || Task 2 loss: %s',
                    epoch_eval['loss'], epoch_eval['t1_loss'],
                    epoch_eval['t2_loss'])

    tf.logging.info(
        'eval Task 1 mindiff loss: %s || eval Task 2 mindiff loss: %s',
        epoch_eval['t1_mindiff'], epoch_eval['t2_mindiff'])

    tf.logging.info('Task 1 Fairness Gap: %s ||Task 2 Fairness Gap: %s',
                    t1_fairness_gap, t2_fairness_gap)

    tf.logging.info('Task 1 Accu: %s || Task 2 Accu: %s', epoch_eval['t1_accu'],
                    epoch_eval['t2_accu'])

    # Track total results.
    total_eval['t1_loss'].append(epoch_eval['t1_loss'].numpy())
    total_eval['t2_loss'].append(epoch_eval['t2_loss'].numpy())
    total_eval['t1_accu'].append(epoch_eval['t1_accu'])
    total_eval['t2_accu'].append(epoch_eval['t2_accu'])
    total_eval['t1_mindiff'].append(epoch_eval['t1_mindiff'].numpy())
    total_eval['t2_mindiff'].append(epoch_eval['t2_mindiff'].numpy())
    total_eval['t1_fairness_gap'].append(t1_fairness_gap)
    total_eval['t2_fairness_gap'].append(t2_fairness_gap)

  return epoch_eval


def hparams():
  return tf.HParams(
      lam_t1=1.0,  # Fairness loss weight for Task 1.
      lam_t2=1.0,
      lam_headratio_t1=1.0,
      lam_headratio_t2=1.0,
      mindiff_type='mmd_mtaf',
      mmd_kernel_decay_length=0.1,
      batch_size=128,
      num_epochs=20,
      use_valid=True,
      lr=0.06,
      alpha=0.5,  # Weight on Task 1.
      shared_units_str='32',
      t1_units_str='16',
      t2_units_str='16',
      t1='is_good_loan',
      t1_type='classification',
      t2='is_high_credit',
      t2_type='classification')


def run_training():
  master_hparams = hparams()

  tf.reset_default_graph()
  final_eval = train(master_hparams)
  tf.logging.info('final results: %s', final_eval)
  print('final results: %s', final_eval)
  return


def main(argv):
  del argv  # Unused.
  run_training()


if __name__ == '__main__':
  app.run(main)
