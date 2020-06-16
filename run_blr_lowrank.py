import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import lazymaps as lm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
plt.close('all')

tfd = tfp.distributions
tfb = tfp.bijectors
DTYPE = tf.float64


# import data
train = pd.read_csv("pd_speech_features.csv")
num_pats = 20
# set up data set (1::3 to use first draw from each person rather than all 3)
y_train = train.values[1::3, -1].astype(float)[:num_pats+1]
x_train = train.values[1::3, 1:501].astype(float)[:num_pats+1,:]
x_train = preprocessing.scale(x_train)

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
        name = 'result_blr_low_rank_problem_notlazy_iaf_sample_size_' + str(sample_size) +\
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
        name = 'result_blr_low_rank_problem_lazy_'+modeltype +\
               '_rank_'+str(lazy_rank) + \
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
        h_is, h_q0 = lm.compute_h_is(base_dist, log_l, 2*num_features, model.transformed_dist.bijector, whitening_bij)
        final_q0_trace = np.trace(h_q0)
        final_is_trace = np.trace(h_is)
        [eigvals, _] = np.linalg.eigh(h_q0)
        eigvals_list.append(eigvals[::-1])
        np.savetxt(name + '_eigvals', np.array(eigvals_list))


        final_elbo = lm.compute_elbo(base_dist, log_l, 2*num_features, model.transformed_dist.bijector, whitening_bij)
        final_var_diag = lm.variance_diag(base_dist, log_l, 2*num_features, model.transformed_dist.bijector, whitening_bij)

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


# U-IAF (rotation, no reduction)
if example_number == 2:
    run_lazy_experiment(lazy_rank=num_features,
                        sample_size=sample_size,
                        num_iters=20000,
                        num_lazy_layers=1,
                        num_trials=num_trials,
                        modeltype='iaf',
                        verbose=False)


# U_r-IAF (rotation and reduction)
if example_number == 3:
    model = run_lazy_experiment(lazy_rank=num_pats,
                                sample_size=sample_size,
                                num_iters=20000,
                                num_lazy_layers=1,
                                num_trials=num_trials,
                                modeltype='iaf',
                                verbose=False)


# plotting histograms Appendix
if False:
    bij = model.bij
    traces_is_id_all = []
    traces_q0_id_all = []
    traces_is_bij_all = []
    traces_q0_bij_all = []
    #
    for ss in [500]:

        traces_is_id = []
        traces_q0_id = []
        traces_is_bij = []
        traces_q0_bij = []

        for i in range(100):
            print('at sample i = '+str(i))
            H_is, H_q0 = lm.compute_h_is(base_dist, log_l, ss, tfb.Identity(), whitening_bij=whitening_bij)
            traces_is_id.append(np.trace(H_is))
            traces_q0_id.append(np.trace(H_q0))

            H_is, H_q0 = lm.compute_h_is(base_dist, log_l, ss, bij, whitening_bij=whitening_bij)
            traces_is_bij.append(np.trace(H_is))
            traces_q0_bij.append(np.trace(H_q0))


    plt.figure()
    bins=np.histogram(np.hstack((traces_is_id, traces_q0_id)), bins=40)[1] #get the bin edges
    plt.hist(traces_is_id, bins, alpha=0.5, label='tr($H$)')
    plt.hist(traces_q0_id, bins, alpha=0.5, label='tr($H^B$)')
    plt.legend(loc=0,fontsize=18)
    plt.xlabel('Traces',fontsize = 18)
    plt.ylabel('Counts',fontsize = 18)
    plt.yticks(fontsize=18)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('blr_lr_beforetraining_hist.pdf')
    plt.show()

    plt.figure()
    bins=np.histogram(np.hstack((traces_is_bij, traces_q0_bij)), bins=40)[1] #get the bin edges
    plt.hist(traces_is_bij, bins, alpha=0.5, label='tr($H$)')
    plt.hist(traces_q0_bij, bins, alpha=0.5, label='tr($H^B$)')
    plt.legend(loc=0,fontsize=18)
    plt.xlabel('Traces',fontsize = 18)
    plt.ylabel('Counts',fontsize = 18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('blr_lr_aftertraining_hist.pdf')
    plt.show()

