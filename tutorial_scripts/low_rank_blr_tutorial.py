import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import _lazymaps.lazymaps as lm
from sklearn import preprocessing

tfd = tfp.distributions
tfb = tfp.bijectors
tf_double = tf.float64

# import data
train = pd.read_csv("pd_speech_features.csv")
num_pats = 20

# set up data set. 1::3 to use first draw from each patient, since we're only looking at 20 patients,
# they should be different patients
y_train = train.values[1::3, -1].astype(float)[:num_pats]
x_train = train.values[1::3, 1:501].astype(float)[:num_pats,:]
x_train = preprocessing.scale(x_train)

# get data dimensions
num_observations = x_train.shape[0]
num_features = x_train.shape[1]    # + 1 to account for constant

# define base distribution as standard normal
base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([num_features], tf_double))

# define prior as zero mean, with variance
prior_var = 100
prior_std = np.sqrt(prior_var)

# log-likelihood
def log_l(sample):
    z = tf.matmul(x_train, tf.transpose(sample))

    theta_1 = tf.math.softplus(-z)
    theta_2 = tf.math.softplus(z)

    repeat_number = sample.shape[0]
    Y = np.array([y_train for _ in range(repeat_number)]).transpose()

    log_like_terms = -Y * theta_1 + (Y - 1) * theta_2
    return tf.reduce_sum(log_like_terms, 0)


# Un-normalized target density (whitened)
def unnormalized_log_prob(samples):
    result = base_dist.log_prob(samples)
    for sample in [samples]:
         result += log_l(sample*prior_std)
    return result


# IAF hyper-parameters
num_stages = 4
width = 128
depth = 2

# training hyper-parameters
sample_size = 100
num_iters = 20000
optimizer = tf.optimizers.Adam(1e-3)

# example to run
example = 2


if example == 0:
    dim = num_features
    iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)
    bij = iaf_bij

    step_record, time_record, loss_record = lm.train(base_dist,
                                                     bij,
                                                     bij.trainable_variables,
                                                     unnormalized_log_prob,
                                                     optimizer,
                                                     num_iters,
                                                     sample_size)

# U-IAF
if example == 1:
    bij = tfb.Identity()
    dim = num_features

    # iaf bij
    iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)

    # form lazy map
    new_bij = lm.make_lazy_bij(iaf_bij, num_features, dim)

    bij, step_record_layer, time_record_layer, loss_record_layer = lm.update_lazy_layer(bij,
                                                                                        new_bij,
                                                                                        base_dist,
                                                                                        unnormalized_log_prob,
                                                                                        optimizer,
                                                                                        num_iters,
                                                                                        sample_size)
# Ur-IAF
if example == 2:
    bij = tfb.Identity()
    dim = num_observations

    # iaf bij
    iaf_bij = lm.make_iaf_bij(dim=dim, num_stages=num_stages, width=width, depth=depth)

    # form lazy map
    new_bij = lm.make_lazy_bij(iaf_bij, num_features, dim)

    bij, step_record_layer, time_record_layer, loss_record_layer = lm.update_lazy_layer(bij,
                                                                                        new_bij,
                                                                                        base_dist,
                                                                                        unnormalized_log_prob,
                                                                                        optimizer,
                                                                                        num_iters,
                                                                                        sample_size)




var_diag = lm.compute_var_diagnostic_tlp(base_dist, bij, unnormalized_log_prob, int(1e3))
elbo = lm.compute_elbo_tlp(base_dist, bij, unnormalized_log_prob, int(1e3))
h_is, h_q0 = lm.compute_h_diagnostic_tlp(base_dist, bij, unnormalized_log_prob, int(1e3))
h_is_trace = np.trace(h_is)
h_q0_trace = np.trace(h_q0)
print('RAN EXAMPLE: '+str(example))
print('==DIAGNOSTICS==')
print('ELBO: ' + str(elbo.numpy()))
print('Var: ' + str(var_diag.numpy()))
print('trace (IS): ' + str(h_is_trace))
print('trace (q0): ' + str(h_q0_trace))

