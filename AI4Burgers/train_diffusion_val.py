import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, vmap, jit, jacrev
from functools import partial
import jax.random as random
#from jax.experimental import optimizers
import jax.example_libraries.optimizers as optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan
from jax.nn import softplus
from jax import config
from jax.flatten_util import ravel_pytree
import pickle

from scipy.stats import boxcox
from scipy.special import inv_boxcox

import optax
import flax.linen as nn

config.update("jax_enable_x64", True)



beta_min = 0.001
beta_max = 3
n_hidden = 1000

def beta_t(t):
    """
    t: time (number)
    returns beta_t as explained above
    """
    return beta_min + t*(beta_max - beta_min)

def alpha_t(t):
    """
    t: time (number)
    returns alpha_t as explained above
    """
    return t*beta_min + 0.5 * t**2 * (beta_max - beta_min)

def drift(x, t):
    """
    x: location of J particles in N dimensions, shape (J, N)
    t: time (number)
    returns the drift of a time-changed OU-process for each batch member, shape (J, N)
    """
    return -0.5*beta_t(t)*x

def dispersion(t):
    """
    t: time (number)
    returns the dispersion
    """
    return jnp.sqrt(beta_t(t))

def mean_factor(t):
    """
    t: time (number)
    returns m_t as above
    """
    return jnp.exp(-0.5 * alpha_t(t))

def var(t):
    """
    t: time (number)
    returns v_t as above
    """
    return 1 - jnp.exp(-alpha_t(t))


R = 1000
train_ts = jnp.arange(1, R)/(R-1)
#we jit the function, but we have to mark some of the arguments as static,
#which means the function is recompiled every time these arguments are changed,
#since they are directly compiled into the binary code. This is necessary
#since jitted-functions cannot have functions as arguments. But it also 
#no problem since these arguments will never/rarely change in our case,
#therefore not triggering re-compilation.
@partial(jit, static_argnums=[1,2,3,4])
def reverse_sde(rng, N, forward_drift, dispersion, score, masks, ts=train_ts):
    """
    rng: random number generator (JAX rng)
    N: dimension in which the reverse SDE runs
    N_initial: How many samples from the initial distribution N(0, I), number
    forward_drift: drift function of the forward SDE (we implemented it above)
    disperion: dispersion function of the forward SDE (we implemented it above)
    score: The score function to use as additional drift in the reverse SDE
    ts: a discretization {t_i} of [0, T], shape 1d-array
    """
    def f(carry, params):
        t, dt = params
        x, mask, rng = carry
        rng, step_rng = jax.random.split(rng)
        disp = dispersion(1-t)
        t = jnp.ones((x.shape[0], 1)) * t
        drift = -forward_drift(x, 1-t) + disp**2 * score(x, 1-t, mask)
        noise = random.normal(step_rng, x.shape)
        x = x + dt * drift + jnp.sqrt(dt)*disp*noise
        return (x, mask, rng), ()
    
    rng, step_rng = random.split(rng)
    n_samples = len(masks)
    initial = random.normal(step_rng, (n_samples, N))
    dts = ts[1:] - ts[:-1]
    params = jnp.stack([ts[:-1], dts], axis=1)
    (x, _, _), _ = scan(f, (initial, masks, rng), params)
    return x

class ApproximateScore(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""
    n_hidden: int = 256
    @nn.compact
    def __call__(self, x, t, mask):
        in_size = x.shape[1]
        n_hidden = self.n_hidden
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)],axis=1)
        x = jnp.concatenate([x, t, mask],axis=1)
        x = nn.Dense(n_hidden*2)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x

def loss_fn(params, model, rng, batch):
    r"""
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    rng, step_rng = random.split(rng)
    N_batch = batch[0].shape[0]
    t = random.randint(step_rng, (N_batch,1), 1, R)/(R-1)
    mean_coeff = mean_factor(t)
    #is it right to have the square root here for the loss?
    vs = var(t)
    stds = jnp.sqrt(vs)
    rng, step_rng = random.split(rng)
    xt, mask = batch
    noise = random.normal(step_rng, xt.shape)
    xt = xt * mean_coeff + noise * stds
    output = model.apply(params, xt, t, mask)
    loss = jnp.mean((noise + output*vs)**2*mask)
    return loss

@partial(jit, static_argnums=(4,5,))
def update_step(params, rng, batch, opt_state, model, optimizer):
    r"""
    params: the current weights of the model
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    opt_state: the internal state of the optimizer
    model: the score function

    takes the gradient of the loss function and updates the model weights (params) using it. Returns
    the value of the loss function (for metrics), the new params and the new optimizer state
    """
    val, grads = jax.value_and_grad(loss_fn)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state

def train_diffusion(train_data, val_data, model, N_epochs, train_size, batch_size, steps_per_epoch, rng, params, optimizer, opt_state): # model is score_model
    for k in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = []
        for perm in perms:
            batch = train_data[:, perm, :] #this is supposed to be 
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, model, optimizer) 
            losses.append(loss)
        mean_loss = jnp.mean(np.array(losses))

        if k % 1000 == 0:
            gt_vals, gt_mask = val_data
            n_ingr = gt_vals.shape[1]
            trained_score = lambda x, t, mask: model.apply(params, x, t, mask)
            rng, sample_rng = random.split(rng)
            samples = reverse_sde(sample_rng, n_ingr, drift, dispersion, trained_score, gt_mask).block_until_ready()
            val_loss = np.mean((gt_vals - samples)**2*gt_mask)
            print("Epoch %d, \t Train Loss: %f, \t Val Loss: %f" % (k, mean_loss, val_loss))
    return params


def train_diff_val():
    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    n_recipes = len(recipes)
    n_ingr = len(ingr_names)
    gt_mask = jnp.array(recipes>0, dtype='int32')

    recipes_normalized = []
    fitted_lambdas = []
    means_stds = []
    for i in range(n_ingr):
        ingr_i = recipes[:,i]
        nonzero_vals = ingr_i[gt_mask[:,i].astype(bool)]
        transformed, fitted_lambda = boxcox(nonzero_vals)
        mean, std = transformed.mean(), transformed.std()
        transformed = (transformed - mean)/std
        ingr_i[gt_mask[:,i].astype(bool)] = transformed
        recipes_normalized.append(ingr_i)
        fitted_lambdas.append(fitted_lambda)
        means_stds.append([mean, std])
    recipes_normalized = jnp.array(recipes_normalized).T

    data = jnp.array([recipes_normalized, gt_mask])
    rng = random.PRNGKey(2025)
    data = jax.random.permutation(rng, data, axis=1)
    split_point = int(0.8*n_recipes) #80-20 split
    train_data = data[:,:split_point,:]
    val_data = data[:,split_point:,:]

    N_epochs = 20000
    lr = 1e-3

    print(f"Training with: N_epochs = {N_epochs}, lr = {lr}")
    batch_size = 400
    
    # Initialize the score function with some dummy input data of the right shape
    score_model = ApproximateScore(n_hidden=n_hidden) # from diffusion_utils
    params = score_model.init(rng, x = jnp.zeros([batch_size, n_ingr]), t = jnp.ones((batch_size, 1)), mask = jnp.zeros([batch_size, n_ingr]))

    #Initialize the optimizer
    schedule = optax.cosine_decay_schedule(
        init_value=lr,          # starting LR
        decay_steps=N_epochs,   # total steps to decay over
        alpha=0.01              # final LR = alpha * init_value
    )
    # optimizer = optax.adam(5.e-4)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    train_size = n_recipes
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size

    params = train_diffusion(train_data, val_data, score_model, N_epochs, train_size, batch_size, steps_per_epoch, rng, params, optimizer, opt_state)
    with open(f'params/diff_value_params.npy', 'wb') as f:
        pickle.dump([params, fitted_lambdas, means_stds, train_data, val_data], f)

    # Sample using the trained params
    trained_score = lambda x, t, mask: score_model.apply(params, x, t, mask)
    rng, step_rng = random.split(rng)
    samples = reverse_sde(step_rng, n_ingr, drift, dispersion, trained_score, gt_mask).block_until_ready()
    with open(f'params/diff_value_samples.npy', 'wb') as f:
        pickle.dump(samples, f)

if __name__ == "__main__":
    train_diff_val()