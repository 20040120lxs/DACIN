import numpy as np
import tensorflow as tf

def int2bits(x, n, out_dtype=None):
  """Convert an integer x in (...) into bits in (..., n) with values -1 and 1."""
  x = tf.bitwise.right_shift(tf.expand_dims(x, -1), tf.range(n))
  x = tf.math.mod(x, 2)
  x = tf.cast(x, tf.float32) * 2 - 1
  if out_dtype and out_dtype != x.dtype:
    x = tf.cast(x, out_dtype)
  return x


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  #print("labels_dense: ", labels_dense, "num_labels: ", num_labels)
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  #print("labels_one_hot_flat: ", len(list(labels_one_hot.flat)))
  #print("index_offset: ", index_offset, "labels_dense.ravel():", labels_dense.ravel().astype(int))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(int)] = 1
  return labels_one_hot


def normalization(data, parameters=None):
    # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    if parameters is None:
        min_arr = np.nanmin(norm_data, axis=0)  # min by column
        max_arr = np.nanmax(norm_data + 1e-6, axis=0)
        norm_data = np.divide((norm_data - min_arr), (max_arr - min_arr))
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_arr,
                           'max_val': max_arr}
    else:
        min_arr = parameters['min_val']  # min by column
        max_arr = parameters['max_val']
        norm_data = np.divide((norm_data - min_arr), (max_arr - min_arr))
        norm_parameters = parameters
    return norm_data, norm_parameters


def miss_data_gen(ori_data_x, data_m):
    miss_data_x = ori_data_x.copy()
    miss_data_x[data_m == 0] = 0  # replace the MVs with 0, in data_m, 0 means missing, 1 means complete
    return miss_data_x


def rmse_loss(data_x, imp_x, data_m):
    # Only for missing values
    nominator = np.sum(((1 - data_m) * data_x - (1 - data_m) * imp_x) ** 2)
    denominator = np.sum(1 - data_m)
    rmse = np.sqrt(nominator / float(denominator))
    return rmse


def data_m_gen(no, dim):
    miss_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    for miss_rate in miss_ratios:
        print("miss_rate: ", miss_rate)
        data_m = binary_sampler(1 - miss_rate, no, dim)
        data_m_filename = "../data_optdigits/miss_rates/{:.0%}/data_m.txt".format(miss_rate)
        np.savetxt(data_m_filename, data_m, delimiter=",", fmt='%s')


def data_h_gen(data_m, hint_rate, no, dim):
    data_h = binary_sampler(hint_rate, no, dim)
    data_h = data_m * data_h
    return data_h


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

