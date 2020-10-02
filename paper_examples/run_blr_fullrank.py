import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import lazymaps as lm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys

tfd = tfp.distributions
tfb = tfp.bijectors
DTYPE = tf.float64


# import data
train = pd.read_csv("pd_speech_features.csv")

# set up data set

# Using first 500 feautures
all_x = train.values[1:, 1:501].astype(float)
all_x = preprocessing.scale(all_x)
all_y = train.values[1:, -1].astype(float)

# 605 = 80% of observations
num_pats = 605
y_train = all_y[:num_pats+1]
x_train = all_x[:num_pats+1,:]


# get data dimensions
num_observations = x_train.shape[0]
num_features = x_train.shape[1]    # + 1 to account for constant

# define base distribution as standard normal
base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([num_features], DTYPE))

# whitening transform (bijector from standard normal to prior):
scale = tf.constant(10, DTYPE)
whitening_lo = tf.linalg.LinearOperatorScaledIdentity(num_features, scale)
whitening_bij = tfb.ScaleMatvecLinearOperator(whitening_lo)

# define log likelihood model
def log_l(w):
    z = tf.matmul(x_train, tf.transpose(w))

    theta_1 = tf.math.softplus(-z)
    theta_2 = tf.math.softplus(z)

    repeat_number = w.shape[0]
    Y = np.array([y_train for _ in range(repeat_number)]).transpose()

    log_like_terms = -Y * theta_1 + (Y - 1) * theta_2
    return tf.reduce_sum(log_like_terms, 0)


# function for baseline iaf experiments
def run_iaf_experiment(sample_size, num_iters, num_trials=10, verbose=False):
    for trial in range(num_trials):
        print('=======================================')
        name = 'result_blr_full_rank_problem_notlazy_iaf_sample_size_' + str(sample_size) +\
               '_trial_' + str(trial)
        print(name)
        model = lm.IafMap(base_dist=base_dist,
                          log_l=log_l,
                          sample_size=sample_size,
                          num_iters=num_iters,
                          rank=num_features,
                          num_stages=4,
                          depth=2,
                          whitening_bij=whitening_bij,
                          verbose=verbose,
                          step_size=1e-3)

        losses, traces, traces_is, neg_elbos, steps = model.train()

        if verbose:
            training_data = np.array([losses, traces, traces_is, neg_elbos, steps])
            np.savetxt(name + '_training_data', training_data)

        # compute final diagnostics with many samples
        h_is, h_q0 = lm.compute_h_is(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector, whitening_bij)
        final_q0_trace = np.trace(h_q0)
        final_is_trace = np.trace(h_is)

        final_elbo = lm.compute_elbo(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector, whitening_bij)
        final_var_diag = lm.variance_diag(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector,whitening_bij)

        diagnostics = [final_q0_trace, final_is_trace, final_elbo, final_var_diag]
        np.savetxt(name + '_diagnostics', np.array(diagnostics))

    return model

# function for lazy-iaf experiments
def run_lazy_experiment(lazy_rank, sample_size, num_iters, num_lazy_layers, num_trials=10, modeltype='linear', verbose=False):
    for trial in range(num_trials):
        print('=======================================')
        name = 'result_blr_full_rank_problem_lazy_'+modeltype+'_rank_'+str(lazy_rank) + \
               '_num_lazy_layers_'+str(num_lazy_layers) + \
               '_sample_size_' + str(sample_size) + \
               '_trial_'+str(trial)

        print(name)

        model = lm.LazyMap(base_dist=base_dist,
                           num_lazy_layers=num_lazy_layers,
                           rank=lazy_rank,
                           num_iters=num_iters,
                           log_l=log_l,
                           whitening_bij=whitening_bij,
                           h_sample_size=num_features,
                           sample_size=sample_size,
                           num_stages=4,
                           depth=2,
                           act=modeltype,
                           verbose=verbose,
                           step_size=1e-3)

        losses, traces, traces_is, neg_elbos, steps, eigvals_list = model.train()

        if verbose:
            training_data = np.array([losses, traces, traces_is, neg_elbos, steps])
            np.savetxt(name + '_training_data', training_data)

        # compute final diagnostics with many samples
        h_is, h_q0 = lm.compute_h_is(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector, whitening_bij)
        final_q0_trace = np.trace(h_q0)
        final_is_trace = np.trace(h_is)
        [eigvals, _] = np.linalg.eigh(h_q0)
        eigvals_list.append(eigvals[::-1])
        np.savetxt(name + '_eigvals', np.array(eigvals_list))

        final_elbo = lm.compute_elbo(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector, whitening_bij)
        final_var_diag = lm.variance_diag(base_dist, log_l, 2 * num_features, model.transformed_dist.bijector,whitening_bij)

        diagnostics = [final_q0_trace, final_is_trace, final_elbo, final_var_diag]
        np.savetxt(name + '_diagnostics', np.array(diagnostics))

    return model

# verbose = True computes diagnostics (ELBO, trace and variance diagnostics) every 1000 iterations
# verbose = False will only report training loss
example_number = 1 #int(sys.argv[1])
sample_size = 100
num_trials = 10

# Baseline IAF
if example_number == 1:
    run_iaf_experiment(sample_size=sample_size,
                       num_iters=20000,
                       num_trials=num_trials,
                       verbose=False)


# U-IAF (rotation)
if example_number == 2:
    run_lazy_experiment(lazy_rank=num_features,
                        sample_size=sample_size,
                        num_iters=20000,
                        num_lazy_layers=1,
                        num_trials=num_trials,
                        modeltype='iaf',
                        verbose=False)


# G3-IAF, 3-lazy layers
if example_number == 3:
    run_lazy_experiment(lazy_rank=200,
                        sample_size=sample_size,
                        num_iters=[5000,5000,10000],
                        num_lazy_layers=3,
                        num_trials=num_trials,
                        modeltype='iaf',
                        verbose=False)

