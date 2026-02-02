# bitflip_diffusion_flax.py
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from functools import partial
import pickle

# -----------------------------
# Config
# -----------------------------
D = 146          # sequence length
T = 1000        # diffusion steps
K = 2           # number of categories
BATCH_SIZE = 1000
TRAIN_STEPS = 100_000
LR = 5e-4
KEY = jax.random.PRNGKey(0)

# Diffusion schedule: linear
betas = jnp.linspace(1e-4, 0.005, T)         # (T,)
alphas = 1.0 - betas                          # for bit-flip
alpha_bars = jnp.cumprod(alphas)            # (T,)

# -----------------------------
# Forward (q) process helpers
# -----------------------------
# def q_marginal_prob(x0, t):
#     """
#     Compute Bernoulli prob for x_t=1 given x0 and t.
#     """
#     alpha_bar_t = alpha_bars[t-1]  # scalar
#     uniform = jnp.ones(K) / K
#     return alpha_bar_t * x0 + (1-alpha_bar_t) * uniform
def q_marginal_prob(x0, t):
    """
    Compute the forward noising step    q(x_t|x_0)          (Eqn. 12)
    We don't need to do this step by step, Eq. 12 allows us to jump straight from x_0 to x_t.
    """
    alpha_bar_t = alpha_bars[t-1]
    """
    Ordinarily, we would return alpha_bar_t * x0 + (1-alpha_bar_t) * uniform with uniform = jnp.ones(K)/K (See Eq. 12)
    (where x0 is one-hot encoded).
    What this does is: 
        - probability of x0 being reselected = alpha_bar_t + (1-alpha_bar_t)/K
        - Probability of any other index being randomly selected = (1-alpha_bar_t)/K
    But this form requires one-hit encoding everything.
    In our case, there are only 2 categories, which we can think of as 0 and 1. So we need to return just 1 float for every point.
    This float is the probability of x0 being reselected.
    The following returns:
        - (alpha_bar_t + (1-alpha_bar_t)/K) if x0 = 1
        - (1-alpha_bar_t)/K if x0 = 0
    achieving the same purpose.
    """
    return (alpha_bar_t + (1-alpha_bar_t)/K) * x0 + (1-alpha_bar_t)/K * (1.0-x0)

q_marginal_vmap_over_batch = jax.vmap(q_marginal_prob, in_axes=(0,0))
q_marginal_vmap_over_posit = jax.vmap(q_marginal_vmap_over_batch, in_axes=(1,None))

def q_sample(rng, x0, t):
    """
    Sample x_t ~ q(x_t | x0).
    """
    # p1 = q_marginal_prob(x0, t)
    p1 = q_marginal_vmap_over_posit(x0, t).T
    return jax.random.bernoulli(rng, p1).astype(jnp.int32)

# -----------------------------
# Model
# -----------------------------
class neuralnet(nn.Module):
    hidden_dim: int = 512
    emb_dim: int = 128

    @nn.compact
    def __call__(self, x_t, t):
        h_x = nn.Dense(self.hidden_dim)(x_t)
        h_x = nn.relu(h_x)

        # Embed timestep
        h_t = nn.Embed(num_embeddings=1001, features=self.emb_dim)(t)
        h_t = nn.Dense(self.emb_dim)(h_t)
        h_t = nn.relu(h_t)

        # Combine x and t embeddings
        h = jnp.concatenate([h_x, h_t], axis=-1)

        # Hidden layers
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)

        # Output layer -> logits for K categories
        logits = nn.Dense(D)(h)
        return logits

# -----------------------------
# Training state
# -----------------------------
class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, lr):
    model = neuralnet()
    params = model.init(rng, jnp.ones((1,D), jnp.int32), jnp.ones((1,), jnp.int32))["params"]
    tx = optax.adam(lr)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# -----------------------------
# Loss
# -----------------------------
def loss_fn(params, rng, x0, t):
    x_t = q_sample(rng, x0, t)
    logits = neuralnet().apply({"params": params}, x_t, t) # estimated x0_hat logits
    loss = optax.sigmoid_binary_cross_entropy(logits, x0).mean()
    return loss

@jax.jit
def train_step(state, rng, x0, t):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, rng, x0, t)
    state = state.apply_gradients(grads=grads)
    return state, loss

# -----------------------------
# Sampler (ancestral)
# -----------------------------
def posterior_step(rng, params, x_t, t):
    """
    One step of ancestral sampling: sample x_{t-1} given x_t.

    Note that q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}) * q(x_{t-1}|x_0)/q(x_t|x_0) (just before Eq. 3)
    where 
        q(x_t|x_{t-1}) is one step of the forward noising process (i.e. the categorical kernel) (Eq. 11)
        q(x_{t-1}|x_0) is the closed form inverse (Eq. 12)
        q(x_t|x_0) is just the normalization factor that is calculated by summing over the product of the two above
    """
    # Predict p(x0|x_t)
    t_vec = jnp.broadcast_to(t, (len(x_t),))
    logits = neuralnet().apply({"params": params}, x_t, t_vec)
    p_x0 = jax.nn.sigmoid(logits)  # (B,D) # Ordinarily, this should have been a softmax. But since we have only 2 categories, we only need 1 float [0,1] for each
    x0_hat = jax.random.bernoulli(rng, p_x0).astype(jnp.int32)

    # Compute posterior for x_{t-1} given (x_t, p_x0)
    beta_t = betas[t]
    alpha_bar_t = alpha_bars[t]
    alpha_bar_tm1 = alpha_bars[t-1] if t > 0 else 1.0

    """
    Since we have only 2 categories: 0 and 1, I have decided to avoid one-hot encoding, and I am simplifying
    categorical distributions with bernoulli distributions. In the paper, they return the probability of each
    category being selected, but I return a single number: the probability of category 1 getting selected.

    But now, for q(x_{t-1}|x_t, x_0) in order to calculate the probability of 1 getting selected, we need to
    first calculate the unnormalized probability, then normalize it by summing over categories, i.e., sum 
    over 0 and 1. To do this, I will calculate the probability of 1 and zero separately. (The zero case is
    denoted with aux)
    """
    q_x_t_given_x_tm1 = (1 - beta_t + beta_t/K) * x_t + beta_t/K * (1 - x_t)
    q_x_tm1_given_x0 = (alpha_bar_tm1 + (1-alpha_bar_tm1)/K) * x0_hat + (1-alpha_bar_tm1)/K * (1 - x0_hat)

    aux_1 = (1 - beta_t + beta_t/K) * (1-x_t) + beta_t/K * x_t
    aux_2 = (alpha_bar_tm1 + (1-alpha_bar_tm1)/K) * (1-x0_hat) + (1-alpha_bar_tm1)/K * x0_hat

    unnormalized = q_x_tm1_given_x0 * q_x_t_given_x_tm1
    normalization = unnormalized + aux_1*aux_2
    probs = unnormalized / normalization

    return jax.random.bernoulli(rng, probs).astype(jnp.int32)

def sample(rng, params, num_samples=4):
    x = jax.random.bernoulli(rng, 0.5, (num_samples, D)).astype(jnp.int32) # x_T ~ Bern(0.5)
    for t in range(T,0,-1):
        rng, step_rng = jax.random.split(rng)
        x = posterior_step(step_rng, params, x, t)
    return x

# -----------------------------
# Toy dataset (random sequences)
# -----------------------------
def toy_data(key, num=10000):
    # Balanced random binary sequences
    return jax.random.bernoulli(key, 0.5, (num,D)).astype(jnp.int32)

# -----------------------------
# Main training loop
# -----------------------------
def main():
    rng = jax.random.PRNGKey(42)
    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    gt_mask = jnp.array(recipes>0, dtype='int32')
    n_recipes = len(recipes)
    split_point = int(0.8*n_recipes)
    trn_data = gt_mask[:split_point]
    
    schedule = optax.cosine_decay_schedule(
        init_value=1e-3,    # starting LR
        decay_steps=TRAIN_STEPS, # total steps to decay over
        alpha=0.01          # final LR = alpha * init_value
    )
    state = create_train_state(rng, schedule)
    
    print("Starting training...")
    loss_hist = []
    for step in range(TRAIN_STEPS):
        rng, step_rng, t_rng, batch_rng = jax.random.split(rng, 4)
        # idx = jax.random.randint(batch_rng, (BATCH_SIZE,), 0, gt_mask.shape[0])
        # x0 = gt_mask[idx]
        # t = jax.random.randint(t_rng, (BATCH_SIZE,), 1, T+1)  # in [1,T]

        # Try no minibatching
        x0 = jax.random.permutation(batch_rng, trn_data)
        t = jax.random.randint(t_rng, (x0.shape[0],), 1, T+1)

        state, loss = train_step(state, step_rng, x0, t)
        loss_hist.append(loss)
        if step % 1000 == 0:
            print(f"Step {step}, loss {loss:.4f}")
    with open(f'params/bitflip_mask_params.npy', 'wb') as f:
        pickle.dump([state.params, loss_hist], f)

    # Sampling
    rng, samp_rng = jax.random.split(rng)
    samples = sample(samp_rng, state.params, num_samples=20000)
    print("Generated samples:")
    for item in samples[:100]:
        print(item)
    with open(f'params/bitflip_mask_samples.npy', 'wb') as f:
        pickle.dump(np.array(samples), f)

if __name__ == "__main__":
    main()
