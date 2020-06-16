import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import lazymaps as lm
import tensorflow_probability as tfp
from sklearn import preprocessing
import sys
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
dtype = tf.float64
tf.keras.backend.set_floatx(
    'float64'
)

# load dataset:

np.random.seed(112233)

fx = open('yacht_hydrodynamics.data', 'r')
x = []
y = []
for line in fx:
    line = line.rsplit()
    x.append(line[:6])
    y.append(line[6])
fx.close()
x = np.array(x, dtype=float)
x = preprocessing.scale(x)
y = np.array(y, dtype=float)
y = preprocessing.scale(y)


M = x.shape[0]
m = int(np.ceil(.8*x.shape[0]))

x_train, y_train = x[:m], y[:m]
x_test, y_test = x[m:], y[m:]


input_dim = 6
output_dim = 1
hidden_dim = 20

lam = 0.01
gamma = np.exp(0)

# Non-Bayesian NN training to quantify MAP
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
l2reg = tf.keras.regularizers.l2
input = tfkl.Input(shape=(input_dim,))

NN = tfk.Sequential([tfkl.Dense(hidden_dim,
                                activation='sigmoid',
                                kernel_regularizer=l2reg(l=lam/m),
                                bias_regularizer=l2reg(l=lam/m)),

                    tfkl.Dense(hidden_dim,
                               activation='sigmoid',
                               kernel_regularizer=l2reg(l=lam/m),
                               bias_regularizer=l2reg(l=lam/m)),


                    tfkl.Dense(output_dim,
                               activation='linear',
                               kernel_regularizer=l2reg(l=lam/m),
                               bias_regularizer=l2reg(l=lam/m)),
])


determ_model = tfk.models.Model(input, NN(input))
determ_model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

history_callback = determ_model.fit(x_train, y_train,
                             epochs=1000, batch_size=20,
                             validation_data=(x_test, y_test), )

mse_train = history_callback.history["mean_squared_error"]
losses_train = history_callback.history["loss"]
mse_test = history_callback.history["val_mean_squared_error"]
losses_test = history_callback.history["val_loss"]

# for plotting training and test lost and MSE
#plt.figure()
#plt.plot(losses_train,label='training loss')
#plt.plot(losses_test,label='test loss')
#plt.legend()
#plt.show()
#plt.figure()
#plt.plot(mse_train,label='training mse')
#plt.plot(mse_test,label='test mse')
#plt.legend()
#plt.show()

parameter_dim = determ_model.count_params()
print('Problem dim: ' + str(parameter_dim))

# function to return biases b1,b2,b3 and weights w1,w2,w3 from vector theta of dimension
# parameter_dim
def set_params(theta):
    if theta.shape.__len__() == 1:
        theta = tf.reshape(theta, (1, parameter_dim))

    batch_size = theta.shape[0]

    # pull out biases of length hidden_dim, hidden_dim and output_dim
    b1 = theta[:, :hidden_dim]
    b2 = theta[:, hidden_dim:hidden_dim+hidden_dim]
    b3 = theta[:, hidden_dim+hidden_dim:hidden_dim+hidden_dim+output_dim]

    # pull out weight w1 (hidden_dim X input_dim)
    start_idx = hidden_dim+hidden_dim+output_dim
    end_idx = start_idx + hidden_dim*input_dim
    w1 = theta[:,start_idx:end_idx]
    w1 = tf.reshape(w1, (batch_size, input_dim, hidden_dim))

    # pull out weight w2 (hidden_dim X hidden_dim)
    start_idx = end_idx
    end_idx = start_idx + hidden_dim*hidden_dim
    w2 = theta[:,start_idx:end_idx]
    w2 = tf.reshape(w2, (batch_size, hidden_dim, hidden_dim))

    # pull out weight w3 (output_dim X hidden_dim)
    start_idx = end_idx
    end_idx = start_idx + hidden_dim*output_dim
    w3 = theta[:,start_idx:end_idx]
    w3 = tf.reshape(w3, (batch_size, hidden_dim, output_dim))

    return b1, b2, b3, w1, w2, w3

# function to apply NN for a sample theta
def nn_fn(theta, x):
    if x.shape == (input_dim,):
        x = x.reshape(1,input_dim)

    b1, b2, b3, w1, w2, w3 = set_params(theta)

    L1 = tf.sigmoid(tf.matmul(x,w1) + b1)
    L2 = tf.sigmoid(tf.matmul(L1, w2) + b2)
    L3 = tf.matmul(L2,w3) + b3

    return L3


# standard Gaussian base distribution
base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(parameter_dim, dtype))

# whitening transform (bijector from standard normal to prior):
whitening_lo = tf.linalg.LinearOperatorScaledIdentity(parameter_dim, tf.constant(1 / np.sqrt(lam), dtype))
whitening_bij = tfb.ScaleMatvecLinearOperator(whitening_lo)

# function to compute square error (part of likelihood)
def compute_sq_loss(theta):

    y_pred = tf.reshape(nn_fn(theta, x_train), (m,))

    return tf.reduce_sum(tf.square(y_train - y_pred))

# normalizing constant from likelihood
C1 = 0.5*m*(np.log(gamma/(2*np.pi)))

# define likelihood function
def log_l(theta):
    if theta.shape.__len__() == 1:
        theta = tf.reshape(theta, (1, parameter_dim))

    ll = []
    for t in theta:
        ll.append(C1 - .5*gamma*compute_sq_loss(t))

    return tf.convert_to_tensor(ll)

# Baseline affine experiment
def run_linear_experiment(sample_size, num_iters, num_trials=10, verbose=False):
    for trial in range(num_trials):
        print('=======================================')
        name = 'result_bnn_notlazy_linear_sample_size_' + str(sample_size) +\
               '_trial_' + str(trial)
        print(name)
        model = lm.LinearMap(base_dist=base_dist,
                          log_l=log_l,
                          sample_size=sample_size,
                          num_iters=num_iters,
                          rank=parameter_dim,
                          whitening_bij=whitening_bij,
                          verbose=verbose,
                          step_size=1e-3)

        losses, traces, traces_is, neg_elbos, steps = model.train()

        if verbose:
            training_data = np.array([losses, traces, traces_is, neg_elbos, steps])
            np.savetxt(name + '_training_data', training_data)

        # compute final diagnostics with many samples
        h_is, h_q0 = lm.compute_h_is(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector, whitening_bij)
        final_q0_trace = np.trace(h_q0)
        final_is_trace = np.trace(h_is)

        final_elbo = lm.compute_elbo(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector, whitening_bij)
        final_var_diag = lm.variance_diag(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector,whitening_bij)

        diagnostics = [final_q0_trace, final_is_trace, final_elbo, final_var_diag]
        np.savetxt(name + '_diagnostics', np.array(diagnostics))

    return model


# Lazy experiment
def run_lazy_experiment(lazy_rank, sample_size, num_iters, num_lazy_layers, num_trials=10, modeltype='linear', verbose=False):
    for trial in range(num_trials):
        print('=======================================')
        name = 'result_bnn_lazy_'+modeltype+'_rank_'+str(lazy_rank) + \
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
                           h_sample_size=parameter_dim,
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
        h_is, h_q0 = lm.compute_h_is(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector, whitening_bij)
        final_q0_trace = np.trace(h_q0)
        final_is_trace = np.trace(h_is)
        [eigvals, _] = np.linalg.eigh(h_q0)
        eigvals_list.append(eigvals[::-1])
        np.savetxt(name + '_eigvals', np.array(eigvals_list))

        final_elbo = lm.compute_elbo(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector, whitening_bij)
        final_var_diag = lm.variance_diag(base_dist, log_l, 2 * parameter_dim, model.transformed_dist.bijector,whitening_bij)

        diagnostics = [final_q0_trace, final_is_trace, final_elbo, final_var_diag]
        np.savetxt(name + '_diagnostics', np.array(diagnostics))

    return model


# verbose = True computes diagnostics (ELBO, trace and variance diagnostics) every 1000 iterations
# verbose = False will only report training loss
example_number = 1
sample_size = 100
num_trials = 10

# Baseline affine
if example_number == 1:
    run_linear_experiment(sample_size=sample_size,
                       num_iters=20000,
                       num_trials=num_trials,
                       verbose=False)


# G3 - Affine (3 lazy layers)
if example_number == 3:
    run_lazy_experiment(lazy_rank=200,
                        sample_size=sample_size,
                        num_iters=[5000,5000,10000],
                        num_lazy_layers=3,
                        num_trials=num_trials,
                        modeltype='linear',
                        verbose=False)
