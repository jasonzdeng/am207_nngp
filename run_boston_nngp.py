r"""Run boston home prices regression with NNGP Kernel.

Usage:

python run_boston_experiments.py \
      --depth=10 \
      --verbose=False

"""

import sys
sys.path.append('nngp')

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from nngp.nngp import NNGPKernel
from nngp.gpr import GaussianProcessRegression

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('depth', 10,
                     'Number of hidden layers.')
flags.DEFINE_boolean('verbose', False,
                     'Log tensorflow information messages.')

if FLAGS.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

sys.path.append('nngp')
boston = load_boston()

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=444, test_size=.25)
# scale each predictor to be zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = y_train.reshape(-1, 1)
# don't leak into test data
X_test = scaler.transform(X_test)
y_test = y_test.reshape(-1, 1)

with tf.Session() as sess:
    nngp_kernel = NNGPKernel(
        depth=FLAGS.depth,
        weight_var=1.79,
        bias_var=0.83,
        nonlin_fn=tf.tanh,
        grid_path='grid',
        use_precomputed_grid=True,
        n_gauss=501,
        n_var=501,
        n_corr=501,
        max_gauss=10,
        max_var=100,
        use_fixed_point_norm=False)

    model = GaussianProcessRegression(X_train, y_train, kern=nngp_kernel)
    y_hat_train, train_eps = model.predict(X_train, sess)
    y_hat_test, test_eps = model.predict(X_test, sess)

print('Training MSE: %.3f, R2: %.3f' % (mean_squared_error(y_train, y_hat_train), r2_score(y_train, y_hat_train)))

print('Test MSE: %.3f, R2: %.3f' % (mean_squared_error(y_test, y_hat_test), r2_score(y_test, y_hat_test)))
