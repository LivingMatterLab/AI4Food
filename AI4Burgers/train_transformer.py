from typing import Any, Callable, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from jax import lax
import pickle
import argparse

pos_weight = 1.0

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def make_causal_mask(seq_len: int):
    """Return a (1, 1, seq_len, seq_len) boolean causal mask for Flax SelfAttention.
       True indicates positions that are allowed (i.e. not masked)."""
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

    # Flax SelfAttention expects shape (batch, heads, qlen, klen) or broadcastable.
    return causal  # we will expand / broadcast as needed


def gaussian_logpdf(x: jnp.ndarray, mu: jnp.ndarray, log_sigma: jnp.ndarray):
    """Compute log N(x | mu, sigma) per component: returns (..., K)"""
    # x: (..., 1) or (..., )
    # mu/log_sigma: (..., K)
    x = jnp.expand_dims(x, axis=-1)  # (..., 1)
    var = jnp.exp(2.0 * log_sigma)
    logp = -0.5 * ((x - mu) ** 2) / var - log_sigma - 0.5 * jnp.log(2 * jnp.pi)
    return logp  # (..., K)


def mixture_nll(y: jnp.ndarray, pi_logits: jnp.ndarray, mu: jnp.ndarray, log_sigma: jnp.ndarray):
    """Negative log-likelihood per position for a Gaussian mixture.
       y: (B, D)
       pi_logits, mu, log_sigma: (B, D, K)
       returns nll: (B, D) (per position)"""
    log_pi = jax.nn.log_softmax(pi_logits, axis=-1)  # (B,D,K)
    log_comp = gaussian_logpdf(y, mu, log_sigma)     # (B,D,K)
    log_prob = jax.scipy.special.logsumexp(log_pi + log_comp, axis=-1)  # (B,D)
    return -log_prob



# ---------------------------------------------------------------------
# Transformer blocks (simple)
# ---------------------------------------------------------------------
class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, causal_mask, deterministic: bool = True):
        # Self-attention
        att = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=True,
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.dropout_rate,
        )(x, mask=causal_mask)
        x = x + att
        y = nn.LayerNorm()(x)
        # MLP
        mlp = nn.Dense(self.mlp_dim)(y)
        mlp = nn.gelu(mlp)
        mlp = nn.Dropout(rate=self.dropout_rate)(mlp, deterministic=deterministic)
        mlp = nn.Dense(self.d_model)(mlp)
        x = x + mlp
        x = nn.LayerNorm()(x)
        return x


class ARTransformer(nn.Module):
    """Stacked causal transformer encoder for autoregressive modeling."""
    D: int
    d_model: int = 160 # This has to be divisible by n_heads
    n_layers: int = 6
    n_heads: int = 8
    mlp_dim: int = 1024
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, token_emb: jnp.ndarray, deterministic: bool = True):
        # token_emb: (B, D, d_model)
        seq_len = token_emb.shape[1]
        causal = make_causal_mask(seq_len)  # (D, D)
        # Flax SelfAttention expects mask shape (batch, heads, qlen, klen) or (batch, qlen, klen)
        # We'll broadcast causal across batch and heads in the attention call.
        # Expand to (1, 1, D, D)
        causal_mask = causal.reshape(1, 1, seq_len, seq_len)
        x = token_emb
        for _ in range(self.n_layers):
            x = TransformerBlock(self.d_model, self.n_heads, self.mlp_dim, self.dropout_rate)(
                x, causal_mask, deterministic=deterministic
            )
        return x  # (B, D, d_model)



# ---------------------------------------------------------------------
# Mask AR model (outputs logits for Bernoulli)
# ---------------------------------------------------------------------
class MaskAR(nn.Module):
    D: int
    d_model: int = 160 # This has to be divisible by n_heads
    n_layers: int = 6
    n_heads: int = 8

    @nn.compact
    def __call__(self, m_in: jnp.ndarray, deterministic: bool = True):
        """
        m_in: (B, D) binary teacher-forced inputs (0/1); during training teacher forcing used;
              During evaluation you may pass the previously sampled prefix as m_in too.
        returns logits: (B, D)
        """
        B, D = m_in.shape
        assert D == self.D
        # index embedding
        idx = jnp.arange(D)
        idx_emb = nn.Embed(num_embeddings=self.D, features=self.d_model, name="idx_embed")(idx) # (D, d_model)
        idx_emb = jnp.broadcast_to(idx_emb[None, :, :], (B, D, self.d_model))  # (B, D, d_model)

        # previous-mask scalar projection (treat m_in as a float sequence)
        m_proj = nn.Dense(self.d_model, name="m_proj")(m_in[..., None])  # (B, D, d_model)

        token = idx_emb + m_proj
        h = ARTransformer(D=self.D, d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads)(token, deterministic=deterministic)
        logits = nn.Dense(1, name="out")(h).squeeze(-1)  # (B, D)
        return logits
    

# ---------------------------------------------------------------------
# Value AR model (conditioned on mask; Mixture of Gaussians head)
# ---------------------------------------------------------------------
class MoGHead(nn.Module):
    K: int = 8
    @nn.compact
    def __call__(self, h):
        # h: (B, D, d_model)
        B, D, _ = h.shape
        out = nn.Dense(self.K * 3)(h)  # logits, mu, log_sigma
        out = out.reshape(B, D, self.K, 3)
        pi_logits = out[..., 0]        # (B, D, K)
        mu = out[..., 1]               # (B, D, K)
        log_sigma = out[..., 2]        # (B, D, K)
        # optional clamp on log_sigma numerically
        log_sigma = jnp.clip(log_sigma, -10.0, 5.0)
        return pi_logits, mu, log_sigma


class ValueAR(nn.Module):
    D: int
    d_model: int = 160 # This has to be divisible by n_heads
    n_layers: int = 8
    n_heads: int = 8
    K: int = 8

    @nn.compact
    def __call__(self, v_in: jnp.ndarray, m: jnp.ndarray, deterministic: bool = True):
        """
        v_in: (B, D) floats with zeros where inactive. (teacher-forced during training)
        m: (B, D) binary mask (0/1)
        returns mixture params: pi_logits, mu, log_sigma with shapes (B, D, K)
        """
        B, D = v_in.shape
        assert D == self.D
        idx = jnp.arange(D)
        idx_emb = nn.Embed(num_embeddings=self.D, features=self.d_model, name="idx_embed")(idx)  # (D, d_model)
        idx_emb = jnp.broadcast_to(idx_emb[None, :, :], (B, D, self.d_model))

        mask_emb = nn.Embed(num_embeddings=2, features=self.d_model, name="mask_embed")(m.astype(jnp.int32))  # (B, D, d_model)
        val_proj = nn.Dense(self.d_model, name="val_proj")(v_in[..., None])  # (B,D,d_model)

        token = idx_emb + mask_emb + val_proj
        h = ARTransformer(D=self.D, d_model=self.d_model, n_layers=self.n_layers, n_heads=self.n_heads)(token, deterministic=deterministic)
        pi_logits, mu, log_sigma = MoGHead()(h)
        return pi_logits, mu, log_sigma


# ---------------------------------------------------------------------
# Train state wrappers
# ---------------------------------------------------------------------
class TrainState(train_state.TrainState):
    pass  # add metrics / rng if needed


# ---------------------------------------------------------------------
# Losses and training steps
# ---------------------------------------------------------------------

def mask_loss_fn(mask_logits: jnp.ndarray, m_target: jnp.ndarray):
    bce = optax.sigmoid_binary_cross_entropy(mask_logits, m_target)
    # Compute weights
    # pos_weight = 5.0
    weights = jnp.where(m_target == 1, pos_weight, 1.0)
    weighted_bce = bce * weights
    return weighted_bce.sum(axis=-1).mean()


def value_loss_fn(pi_logits, mu, log_sigma, v_target, m_target, eps: float = 1e-8):
    """Masked mixture NLL. v_target: (B,D), m_target: (B,D) binary (1=active)."""
    nll = mixture_nll(v_target, pi_logits, mu, log_sigma)  # (B,D)
    # zero out positions where m_target == 0
    masked_nll = nll * m_target
    # normalize by number of active positions per example to stabilize loss scale
    active_counts = jnp.sum(m_target, axis=-1)  # (B,)
    # avoid division by zero: for examples with 0 active dims, treat loss as 0
    avg_nll_per_example = jnp.where(active_counts > 0, jnp.sum(masked_nll, axis=-1) / (active_counts + eps), 0.0)
    return avg_nll_per_example.mean()  # scalar

def make_inputs_and_labels(batch_targets, bos_token, eos_token):
    # batch_targets: (batch, T)   with EOS already appended
    batch_size = batch_targets.shape[0]

    bos_column = jnp.full((batch_size, 1), bos_token)
    inputs = jnp.concatenate([bos_column, batch_targets[:, :-1]], axis=1)  # prepend BOS
    labels = batch_targets  # predict original sequence (incl EOS)
    
    return inputs, labels

@jax.jit
def train_step_mask(state: TrainState, batch_m: jnp.ndarray, dummy: jnp.ndarray, rng: jax.random.PRNGKey):
    inputs, labels = make_inputs_and_labels(batch_m, 0.0, 0.0)

    """Single step updating mask model. batch_m: (B,D) binary teacher forced."""
    def loss_fn(params, microbatch):
        microbatch_inputs, microbatch_labels = microbatch
        logits = MaskAR(D=inputs.shape[1]).apply({'params': params}, microbatch_inputs, deterministic=False, rngs={'dropout': rng})
        loss = mask_loss_fn(logits, microbatch_labels)
        return loss, logits
    
    #Implement microbatching, because we need to use a larger batch size but it results in memory issues.
    batch_size = len(batch_m)
    microbatch_size = 100
    microbatches = [(inputs[i:i+microbatch_size], labels[i:i+microbatch_size]) 
                    for i in range(0, batch_size, microbatch_size)]

    loss_list = []
    grads_list = []
    for microbatch in microbatches:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params, microbatch)
        loss_list.append(loss)
        grads_list.append(grads)
    
    # grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads_list)
    grads = jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *grads_list)
    grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, 0.0), grads)
    new_state = state.apply_gradients(grads=grads)
    metrics = {'loss': loss}
    return new_state, metrics


@jax.jit
def train_step_value(state: TrainState, batch_v: jnp.ndarray, batch_m: jnp.ndarray, rng: jax.random.PRNGKey, K: int = 8):
    """Single step updating value model; uses teacher forcing (batch_v contains zeros at inactive positions)."""
    def loss_fn(params):
        pi_logits, mu, log_sigma = ValueAR(D=batch_v.shape[1], K=K).apply({'params': params}, batch_v, batch_m, deterministic=False, rngs={'dropout': rng})
        loss = value_loss_fn(pi_logits, mu, log_sigma, batch_v, batch_m)
        return loss, (pi_logits, mu, log_sigma)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, 0.0), grads)
    new_state = state.apply_gradients(grads=grads)
    metrics = {'loss': loss}
    return new_state, metrics

def create_train_state(rng, model_cls, example_inputs, learning_rate=1e-4):
    model = model_cls
    params = model.init(rng, *example_inputs)['params']
    tx = optax.adamw(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

def train_loop(state, data, aux, num_epochs, batch_size, rng_key, step_fn, print_freq = 100):
    """
    state: your TrainState
    data:  (N_samples, D) jnp.array or np.array of binary data
    """
    metrics_history = []

    for epoch in range(num_epochs):
        # Shuffle dataset each epoch
        rng_key, shuffle_key = jax.random.split(rng_key)
        batch_idx = jax.random.choice(shuffle_key, jnp.arange(len(data)), shape=(batch_size,))
        batch = data[batch_idx]
        aux_batch = aux[batch_idx] #aux is None for mask model and the mask for value model

        rng_key, step_key = jax.random.split(rng_key)
        state, metrics = step_fn(state, batch, aux_batch, step_key)
        metrics_history.append(metrics)

        if epoch % print_freq == 0:
            print(f'Epoch {epoch+1}, loss = {metrics['loss']:.4f}')

    return state, metrics_history

def sample_mask(params, rng: jax.random.PRNGKey, D: int, batch_size: int = 1,
                temperature: float = 1.0, bias: float = 0.0):
    """Sequentially sample mask autoregressively. Returns (B, D) binary array."""
    def sample_one(rng_single):
        """Sample a single mask of length D."""
        def step(carry, i):
            m, rng_i = carry
            inp = m[None, :]  # (1, D)
            logits = MaskAR(D=D).apply({'params': params}, inp, deterministic=True)  # (1, D)
            logit_i = logits[0, i] / temperature + bias

            rng_i, sub = jax.random.split(rng_i)
            m_i = jax.random.bernoulli(sub, jax.nn.sigmoid(logit_i)).astype(jnp.int32)
            m = m.at[i].set(m_i)
            return (m, rng_i), None

        m0 = jnp.zeros((D,), dtype=jnp.int32)
        (m_final, _), _ = lax.scan(step, (m0, rng_single), jnp.arange(D))
        return m_final

    rngs = jax.random.split(rng, batch_size)
    ms = jax.vmap(sample_one)(rngs)
    return ms

def sample_values(value_params, mask, rng: jax.random.PRNGKey, K: int = 8, temperature: float = 1.0):
    """Sequentially sample values conditioned on a given mask (mask: (B,D) int).
       Returns values (B,D) floats, zeros where mask==0."""
    B, D = mask.shape
    rngs = jax.random.split(rng, B)

    def sample_one(args):
        rng_single, mask_single = args
        v = jnp.zeros((D,), dtype=jnp.float32)
        for i in range(D):
            if mask_single[i] == 0:
                v = v.at[i].set(0.0)
                continue
            # prepare prefix v (teacher forcing with sampled values so far)
            v_in = v[None, :]  # shape (1, D)
            m_in = mask_single[None, :]  # (1, D)
            pi_logits, mu, log_sigma = ValueAR(D=D, K=K).apply({'params': value_params}, v_in, m_in, deterministic=True)
            # get params for position i
            logits_i = pi_logits[0, i] / temperature
            mu_i = mu[0, i]  # (K,)
            sigma_i = jnp.exp(log_sigma[0, i])
            # sample mixture component
            rng_single, sub1, sub2 = jax.random.split(rng_single, 3)
            comp = jax.random.categorical(sub1, logits_i)
            comp = jnp.asarray(comp, dtype=jnp.int32)
            # sample from chosen Gaussian
            chosen_mu = mu_i[comp]
            chosen_sigma = sigma_i[comp]
            eps = jax.random.normal(sub2)
            samp = chosen_mu + chosen_sigma * eps
            v = v.at[i].set(samp)
        return v

    # outs = jax.vmap(sample_one)((rngs, mask))
    outs = []
    for i in range(B):
        outs.append(sample_one((rngs[i], mask[i])))
    return jnp.array(outs)  # (B, D)



def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Script that takes input.")
    parser.add_argument("--pos_weight", type=float, required=True)
    
    args = parser.parse_args()
    pos_weight = float(args.pos_weight)

    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    n_ingr = len(ingr_names)

    gt_mask = jnp.array(recipes>0, dtype='int32')

    rng = jax.random.PRNGKey(0)
    # toy dims
    D = n_ingr # dimensionality of the data
    batch_size = 1000

    # init mask model state
    rng, m_init_rng = jax.random.split(rng)
    mask_model = MaskAR(D=D)
    schedule = optax.cosine_decay_schedule(
        init_value=5e-4,    # starting LR
        decay_steps=100000, # total steps to decay over
        alpha=0.01          # final LR = alpha * init_value
    )
    mask_state = create_train_state(m_init_rng, mask_model, (gt_mask[:batch_size],), learning_rate=schedule)
    
    # Restart previous run
    # with open('params/mask_params.npy', 'rb') as f:
    #     mask_params, m_metrics_hist = pickle.load(f)
    # mask_state = mask_state.replace(params=mask_params)

    # train mask model
    rng, s1, s2 = jax.random.split(rng, 3)
    print('Training the mask model...')
    mask_state, m_metrics_hist = train_loop(mask_state, gt_mask, jnp.zeros_like(gt_mask), 100_000, batch_size, s1, train_step_mask)
    with open(f'params/mask_params.npy', 'wb') as f:
        pickle.dump([mask_state.params, m_metrics_hist], f)
    

    # Sampling
    sampled_masks = []
    rngs = jax.random.split(rng, num=10)
    for i in range(10):
        sampled_masks.extend(sample_mask(mask_state.params, rngs[i], D, batch_size=5000)) # need to do smaller batches to fit in memory.
    with open(f'params/samples.npy', 'wb') as f:
        pickle.dump(np.array(sampled_masks), f)

    # # init value model state
    # rng, v_init_rng = jax.random.split(rng)
    # value_model = ValueAR(D=D, K=K)
    # value_state = create_train_state(v_init_rng, value_model, (gt_vals[:batch_size], gt_mask[:batch_size]))

    # # train value model
    # print('Training the value model...')
    # value_state, v_metrics_hist = train_loop(value_state, gt_vals, gt_mask, 10000, batch_size, s2, train_step_value)
    # with open('params/value_params.npy', 'wb') as f:
    #     pickle.dump([value_state.params, v_metrics_hist], f)

if __name__ == "__main__":
    main()