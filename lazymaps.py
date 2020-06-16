import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import time

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = tf.float64

def compute_elbo(base_dist, log_l, sample_size, model_bij, whitening_bij):

    x_samples = base_dist.sample(sample_size)

    t_x = model_bij(x_samples)  # forward map of S.N. samples
    n_t_x = whitening_bij(t_x)

    log_like_term = log_l(n_t_x)

    # Log density term
    log_prob_term = log_like_term + base_dist.log_prob(t_x)

    # Jacobian term
    jacobian_term = model_bij.forward_log_det_jacobian(x_samples, event_ndims=1)

    # Add up all terms:
    elbos = log_prob_term + jacobian_term - base_dist.log_prob(x_samples)
    return  tf.reduce_mean(elbos).numpy()


def variance_diag(base_dist, log_l, sample_size, model_bij, whitening_bij):
    x = base_dist.sample(sample_size)

    t_x = model_bij(x)  # forward map of S.N. samples
    n_t_x = whitening_bij(t_x)
    log_like_term = log_l(n_t_x)

    list = base_dist.log_prob(x) - base_dist.log_prob(t_x) - log_like_term \
           - model_bij.forward_log_det_jacobian(x, event_ndims=1)
    return np.cov(list.numpy())


def compute_h_is(base_dist, log_l, sample_size, model_bij, whitening_bij):
    dim = base_dist.event_shape[0]
    h_is = np.zeros([dim, dim])
    h_q0 = np.zeros([dim, dim])
    y_grad_holder = []
    y_holder = []
    for _ in range(sample_size):
        x = base_dist.sample()
        x = tf.reshape(x, [1, dim])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            t_x = model_bij(x)  # forward map of S.N. samples
            n_t_x = whitening_bij(t_x)
            log_like_term = log_l(n_t_x)
            y = log_like_term + base_dist.log_prob(t_x) + model_bij.forward_log_det_jacobian(x, event_ndims=1) \
                - base_dist.log_prob(x)

        y_grad = tape.gradient(y, x).numpy()
        y_grad_holder.append(y_grad)
        y_holder.append(y.numpy())

    y_np = np.array(y_holder).flatten()
    weights = np.exp(y_np - np.max(y_np))
    sum_weights = np.sum(weights)

    for i in range(sample_size):
        h_is += weights[i] * np.outer(y_grad_holder[i], y_grad_holder[i])
        h_q0 += np.outer(y_grad_holder[i], y_grad_holder[i])

    h_is = h_is / sum_weights
    h_q0 = h_q0 / sample_size
    return h_is, h_q0


class LinearMap:

    def __init__(self,
                 base_dist,
                 log_l,
                 num_iters,
                 rank,
                 sample_size=100,
                 whitening_bij=tfb.Identity(),
                 map_before=tfb.Identity(),
                 verbose=False,
                 step_size=1e-2):

        self.base_dist = base_dist
        dtype = base_dist.dtype
        dim = base_dist.event_shape[0]
        self.dim = dim
        self.dtype = dtype
        self.log_l = log_l
        self.num_iters = num_iters
        self.rank = rank
        self.verbose = verbose
        self.sample_size = sample_size
        self.whitening_bij = whitening_bij
        self.map_before = map_before
        self.step_size = step_size

        shift = tf.Variable(tf.zeros(rank, dtype), dtype=dtype)
        initializer = tf.initializers.GlorotUniform()
        scale = tf.Variable(initializer(shape=[rank, rank], dtype=dtype))
        scale_lin_oper = tf.linalg.LinearOperatorLowerTriangular(scale)
        affine_bij = tfb.AffineLinearOperator(scale=scale_lin_oper,
                                              shift=shift)

        lazy_bijector = tfb.Blockwise([affine_bij, tfb.Identity()], [rank, dim - rank])

        transformed_dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=map_before(lazy_bijector))

        training_vars = affine_bij.trainable_variables

        self.transformed_dist = transformed_dist
        self.training_vars = training_vars

    def train(self):
        t_start = time.time()
        sample_size = self.sample_size
        base_dist = self.base_dist
        transformed_dist = self.transformed_dist
        training_vars = self.training_vars
        whitening_bij = self.whitening_bij
        log_l = self.log_l

        optimizer = tf.optimizers.Adam(self.step_size)
        neg_elbos = []
        global_step = []
        np_losses = []
        traces = []
        traces_is = []
        t1 = time.time()
        for i in range(self.num_iters):

            # Updating training info
            if i % 1000 == 0:

                if self.verbose:
                    H_is, H_q0 = compute_h_is(base_dist=base_dist,
                                              log_l=log_l,
                                              model_bij=transformed_dist.bijector,
                                              sample_size=self.dim,
                                              whitening_bij=whitening_bij)

                    traces.append(np.trace(H_q0))

                    traces_is.append(np.trace(H_is))

                    # compute -ELBO with more samples:
                    x_samples = base_dist.sample(self.dim)

                    t_x = transformed_dist.bijector(x_samples)  # forward map of S.N. samples
                    n_t_x = whitening_bij(t_x)

                    log_like_term = log_l(n_t_x)

                    # Log density term
                    log_prob_term = log_like_term + base_dist.log_prob(t_x)

                    # Jacobian term
                    jacobian_term = transformed_dist.bijector.forward_log_det_jacobian(x_samples, event_ndims=1)

                    # Add up all terms:
                    objective = log_prob_term + jacobian_term - base_dist.log_prob(x_samples)
                    neg_elbos.append(-tf.reduce_mean(objective).numpy())

                    print('step: ' + str(i) +
                          ' trace: ' + str('{:.2e}'.format(np.trace(H_q0))) +
                          ' IS trace: ' + str('{:.2e}'.format(np.trace(H_is))) +
                          ' -ELBO:' + str('{:.2e}'.format(neg_elbos[-1])))

            with tf.GradientTape() as tape:
                tape.watch(training_vars)

                # Define loss
                def loss_fn():
                    x_samples = base_dist.sample(sample_size)

                    t_x = transformed_dist.bijector(x_samples)  # forward map of S.N. samples
                    n_t_x = whitening_bij(t_x)

                    log_like_term = log_l(n_t_x)

                    # Log density term
                    log_prob_term = log_like_term + base_dist.log_prob(t_x)

                    # Jacobian term
                    jacobian_term = transformed_dist.bijector.forward_log_det_jacobian(x_samples, event_ndims=1)

                    # Add up all terms:
                    objective = log_prob_term + jacobian_term - base_dist.log_prob(x_samples)
                    return -tf.reduce_mean(objective)

                loss = loss_fn  # loss_fn() for grad and apply_gradients

            optimizer.minimize(loss=loss, var_list=training_vars)

            if i % 1000 == 0:
                global_step.append(i)
                np_losses.append(loss().numpy())
                print('step:  '+str(i)+'  loss:  '+str(np_losses[-1]))  
        t_end = time.time()


        print('Linear map training time (s):' + str(t_end - t_start))
        return np_losses, traces, traces_is, neg_elbos, global_step


class IafMap:
    def __init__(self,
                 base_dist,
                 log_l,
                 num_iters,
                 rank,
                 sample_size=100,                
                 depth=2,
                 num_stages=4,
                 whitening_bij=tfb.Identity(),
                 map_before=tfb.Identity(),
                 verbose=False,
                 kernel_init='glorot_uniform',
                 step_size=1e-2):

        print('Iaf using rank: ' + str(rank) + ' and num_iters: ' + str(num_iters))

        self.base_dist = base_dist
        dtype = base_dist.dtype
        dim = base_dist.event_shape[0]
        self.dim = dim
        self.dtype = dtype
        self.log_l = log_l
        self.num_iters = num_iters
        self.rank = rank
        self.depth = depth
        self.num_stages = num_stages
        self.sample_size = sample_size
        self.whitening_bij = whitening_bij
        self.map_before = map_before
        self.verbose = verbose
        self.kernel_init = kernel_init
        self.step_size = step_size

        bijectors = []
        for i in range(num_stages):
            made = tfb.AutoregressiveNetwork(params=2,
                                             hidden_units=list(np.repeat(128, depth)),
                                             dtype=dtype,
                                             activation='elu',
                                             kernel_initializer=self.kernel_init)

            bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made)))

            # Permute layer
            perm = [i for i in range(rank)]
            perm = perm[::-1]
            bijectors.append(tfb.Permute(permutation=perm))

        
        iaf_bijector = tfb.Chain(list(reversed(bijectors)))
        
        #shift = tf.Variable(tf.zeros(rank, dtype), dtype=dtype)
        #initializer = tf.initializers.GlorotUniform()
        #scale = tf.Variable(initializer(shape=[rank,], dtype=dtype))
        #scale_lin_oper = tf.linalg.LinearOperatorDiag(scale)
        #affine_bij = tfb.AffineLinearOperator(scale=scale_lin_oper, shift=shift)

        #bijector = affine_bij(iaf_bijector)
        bijector = iaf_bijector
        lazy_bijector = tfb.Blockwise([bijector, tfb.Identity()], [rank, dim - rank])

        transformed_dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=map_before(lazy_bijector))

        transformed_dist.sample()
        training_vars = bijector.trainable_variables

        self.transformed_dist = transformed_dist
        self.training_vars = training_vars

    def train(self):
        t_start = time.time()
        sample_size = self.sample_size
        base_dist = self.base_dist
        transformed_dist = self.transformed_dist
        training_vars = self.training_vars
        whitening_bij = self.whitening_bij
        log_l = self.log_l

        optimizer = tf.optimizers.Adam(self.step_size)
        neg_elbos = []
        global_step = []
        np_losses = []
        traces = []
        traces_is = []
        t1 = time.time()
        for i in range(self.num_iters):

            # Updating training info
            if i % 1000 == 0:

                if self.verbose:
                    H_is, H_q0 = compute_h_is(base_dist=base_dist,
                                              log_l=log_l,
                                              model_bij=transformed_dist.bijector,
                                              sample_size=self.dim,
                                              whitening_bij=whitening_bij)

                    traces.append(np.trace(H_q0))

                    traces_is.append(np.trace(H_is))

                    # compute -ELBO with more samples:
                    x_samples = base_dist.sample(self.dim)

                    t_x = transformed_dist.bijector(x_samples)  # forward map of S.N. samples
                    n_t_x = whitening_bij(t_x)

                    log_like_term = log_l(n_t_x)

                    # Log density term
                    log_prob_term = log_like_term + base_dist.log_prob(t_x)

                    # Jacobian term
                    jacobian_term = transformed_dist.bijector.forward_log_det_jacobian(x_samples, event_ndims=1)

                    # Add up all terms:
                    objective = log_prob_term + jacobian_term - base_dist.log_prob(x_samples)
                    neg_elbos.append(-tf.reduce_mean(objective).numpy())

                    print('step: ' + str(i) +
                          ' trace: ' + str('{:.2e}'.format(np.trace(H_q0))) +
                          ' IS trace: ' + str('{:.2e}'.format(np.trace(H_is))) +
                          ' -ELBO:' + str('{:.2e}'.format(neg_elbos[-1])))



            with tf.GradientTape() as tape:
                tape.watch(training_vars)

                # Define loss
                def loss_fn():
                    x_samples = base_dist.sample(sample_size)

                    t_x = transformed_dist.bijector(x_samples)  # forward map of S.N. samples
                    n_t_x = whitening_bij(t_x)

                    log_like_term = log_l(n_t_x)

                    # Log density term
                    log_prob_term = log_like_term + base_dist.log_prob(t_x)

                    # Jacobian term
                    jacobian_term = transformed_dist.bijector.forward_log_det_jacobian(x_samples, event_ndims=1)

                    # Add up all terms:
                    objective = log_prob_term + jacobian_term - base_dist.log_prob(x_samples)
                    return -tf.reduce_mean(objective)

                loss = loss_fn  # loss_fn() for grad and apply_gradients

            optimizer.minimize(loss=loss, var_list=training_vars)

            if i % 1000 == 0:

                global_step.append(i)
                np_losses.append(loss().numpy())
                print('step:  '+str(i)+'  loss:  '+str(loss().numpy())) 

        t_end = time.time()

        print('IAF map training time (s):' + str(t_end - t_start))
        return np_losses, traces, traces_is, neg_elbos, global_step


class LinearOperatorWithDetOne(tf.linalg.LinearOperatorFullMatrix):
    def determinant(self):
        return tf.convert_to_tensor(1, DTYPE)

    def _determinant(self):
        return tf.convert_to_tensor(1, DTYPE)

    def log_abs_determinant(self):
        return tf.convert_to_tensor(0, DTYPE)

    def _log_abs_determinant(self):
        return tf.convert_to_tensor(0, DTYPE)


class LazyMap:
    def __init__(self,
                 base_dist,
                 num_lazy_layers,
                 rank,
                 num_iters,
                 log_l,
                 whitening_bij=tfb.Identity(),
                 map_before=tfb.Identity(),
                 name='lazymap',
                 h_sample_size=200,
                 sample_size=100,
                 num_stages=4,
                 depth=16,
                 act='iaf',
                 verbose=False,
                 kernel_init='glorot_uniform',
                 step_size=1e-2,
                 important_sampling_on=False):

        # setting up iteration list
        # if num_iters = n, then use same num_iter for all layers: num_iters = [n,n,n,...]
        if type(num_iters) is not list:
            num_iters = np.repeat(num_iters, num_lazy_layers)

        # if num_iters = [n1, n2], then use last num_iter for rest of layers: num_iters = [n1, n2, n2,...]
        elif np.shape(num_iters)[0] < num_lazy_layers:
            rest = [num_iters[-1] for _ in range(num_lazy_layers - np.shape(num_iters)[0])]
            num_iters.extend(rest)

        # setting up rank list
        # if rank = r, then use same rank all layers: rank = [r,r,r,...]
        if type(rank) is not list:
            rank = np.repeat(rank, num_lazy_layers)

        # if rank = [r1, r2], then use last rank for rest of layers: rank = [r1, r2, r2,...]
        elif np.shape(rank)[0] < num_lazy_layers:
            rest = [rank[-1] for _ in range(num_lazy_layers - np.shape(rank)[0])]
            rank.extend(rest)

        # Lazymap specific parameters
        self.num_lazy_layers = num_lazy_layers
        self.h_sample_size = h_sample_size
        self.act = act
        self.name = name
        self.map_before = map_before

        # model parameter
        self.base_dist = base_dist
        dtype = base_dist.dtype
        dim = base_dist.event_shape[0]
        self.dim = dim
        self.dtype = dtype
        self.log_l = log_l
        self.num_iters = num_iters
        self.rank = rank      
        self.depth = depth
        self.num_stages = num_stages
        self.sample_size = sample_size
        self.whitening_bij = whitening_bij
        self.verbose = verbose
        self.kernel_init = kernel_init
        self.step_size = step_size
        self.important_sampling_on = important_sampling_on
        self.mean = []
        self.var = []

    def train(self):
        t_start = time.time() 
        act = self.act
        name = self.name

        map_before = self.map_before

        # training diagnostics
        losses_list = []
        steps_list = []
        traces_list = []
        traces_is_list = []
        neg_elbo_list = []
        eigvals_list = []
        for i in range(self.num_lazy_layers):

            print('layer ' + str(i + 1) + ' of ' + str(self.num_lazy_layers))
            print('Using ' + str(self.num_iters[i]) + ' iterations this layer')
            print('Using ' + str(self.rank[i]) + ' for rank this layer')
            h_is, h_q0 = compute_h_is(base_dist=self.base_dist,
                                      log_l=self.log_l,
                                      sample_size=self.h_sample_size,
                                      model_bij=map_before,
                                      whitening_bij=self.whitening_bij)

            print('Trace   :' + str(np.trace(h_q0)))
            print('IS Trace:' + str(np.trace(h_is)))
            if self.important_sampling_on:
                [eigvals, eigvecs] = np.linalg.eigh(h_is)
                eigvecs = eigvecs[:, ::-1]
                eigvals_list.append(eigvals[::-1])

            else:
                [eigvals, eigvecs] = np.linalg.eigh(h_q0)
                eigvecs = eigvecs[:, ::-1]
                eigvals_list.append(eigvals[::-1])

            operator = LinearOperatorWithDetOne(eigvecs)
            affine = tfb.ScaleMatvecLinearOperator(scale=operator)
            map_before = map_before(affine)

            if act == 'iaf':
                model = IafMap(base_dist=self.base_dist,
                                    log_l=self.log_l,
                                    num_iters=self.num_iters[i],
                                    rank=self.rank[i],
                                    sample_size=self.sample_size,
                                    depth=self.depth,
                                    num_stages=self.num_stages,
                                    whitening_bij=self.whitening_bij,
                                    map_before=map_before,
                                    verbose=self.verbose,
                                    kernel_init=self.kernel_init,
                                    step_size=self.step_size)

            elif act == 'linear':
                model = LinearMap(base_dist=self.base_dist,
                                   log_l=self.log_l,
                                   num_iters=self.num_iters[i],
                                   rank=self.rank[i],
                                   sample_size=self.sample_size,
                                   whitening_bij=self.whitening_bij,
                                   map_before=map_before,
                                   verbose=self.verbose,
                                   step_size=self.step_size)

            losses, traces, traces_is, neg_elbo, _ = model.train()
            losses_list.append(losses)
            traces_list.append(traces)
            traces_is_list.append(traces_is)
            neg_elbo_list.append(neg_elbo)

            map_before = model.transformed_dist.bijector
            plt.close('all')

        self.bij = map_before

        losses_all = np.concatenate([losses_list[i] for i in range(self.num_lazy_layers)])
        traces_all = np.concatenate([traces_list[i] for i in range(self.num_lazy_layers)])
        traces_is_all = np.concatenate([traces_is_list[i] for i in range(self.num_lazy_layers)])
        neg_elbos_all = np.concatenate([neg_elbo_list[i] for i in range(self.num_lazy_layers)])
        steps_all = np.arange(np.sum(self.num_iters), step=1000)

        self.transformed_dist = self.bij(self.base_dist)
        t_end = time.time()
        
        print('Total Lazy Training time (s):'+str(t_end - t_start))

        return losses_all, traces_all, traces_is_all, neg_elbos_all, steps_all, eigvals_list
