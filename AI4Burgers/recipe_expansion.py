import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from functools import partial
import pickle
from train_bitflip_diff import *


# Rewrite the posterior_step() function such that it returns raw probabilities, not the bernoulli outcomes.
# Also make the if statement for alpha_bar_tm1 jax-friendly
def posterior_step(rng, params, x_t, t):
    # Predict p(x0|x_t)
    t_vec = jnp.broadcast_to(t, (len(x_t),))
    logits = neuralnet().apply({"params": params}, x_t, t_vec)
    p_x0 = jax.nn.sigmoid(logits)  # (B,D) # Ordinarily, this should have been a softmax. But since we have only 2 categories, we only need 1 float [0,1] for each
    x0_hat = jax.random.bernoulli(rng, p_x0).astype(jnp.int32)

    # Compute posterior for x_{t-1} given (x_t, p_x0)
    beta_t = betas[t]
    alpha_bar_t = alpha_bars[t]
    # alpha_bar_tm1 = alpha_bars[t-1] if t > 0 else 1.0
    alpha_bar_tm1 = jax.lax.cond(
        t > 0,
        lambda _: alpha_bars[t - 1],
        lambda _: 1.0,
        operand=None
    )

    q_x_t_given_x_tm1 = (1 - beta_t + beta_t/K) * x_t + beta_t/K * (1 - x_t)
    q_x_tm1_given_x0 = (alpha_bar_tm1 + (1-alpha_bar_tm1)/K) * x0_hat + (1-alpha_bar_tm1)/K * (1 - x0_hat)

    aux_1 = (1 - beta_t + beta_t/K) * (1-x_t) + beta_t/K * x_t
    aux_2 = (alpha_bar_tm1 + (1-alpha_bar_tm1)/K) * (1-x0_hat) + (1-alpha_bar_tm1)/K * x0_hat

    unnormalized = q_x_tm1_given_x0 * q_x_t_given_x_tm1
    normalization = unnormalized + aux_1*aux_2
    probs = unnormalized / normalization

    # return jax.random.bernoulli(rng, probs).astype(jnp.int32)
    return probs

# Rewrite the sample() function such that it always maintains a value of x=1 for the observed ingredients
def sample_expand(rng, params, observed_mask, num_samples = 1000):
    x = jax.random.bernoulli(rng, 0.5, (num_samples, D)).astype(jnp.int32)  # x_T ~ Bern(0.5)

    def body_fn(t, carry):
        rng, x = carry
        rng, step_rng = jax.random.split(rng)
        probs = posterior_step(step_rng, params, x, t)
        x = jax.random.bernoulli(rng, probs).astype(jnp.int32)
        # Set observed bits to 1
        x = jnp.where(observed_mask == 1, 1, x)
        return (rng, x)

    # Run loop from T down to 1
    rng, x = jax.lax.fori_loop(0, T, lambda i, c: body_fn(T - i, c), (rng, x))

    # Compute final probs for returning
    probs = posterior_step(rng, params, x, 0)
    probs = jnp.where(observed_mask == 1, 1, probs)
    return probs
sample_expand_jit = jax.jit(sample_expand, static_argnums=3)




if __name__ == "__main__":
    # Load the data and the trained params
    with open(f'params/bitflip_mask_params.npy', 'rb') as f:
        params, _ = pickle.load(f)

    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    n_recipes = len(recipes)
    gt_mask = jnp.array(recipes>0, dtype='int32')
    split_point = int(0.8*n_recipes) #80-20 split
    test_data = gt_mask[split_point:]

    out = [] 
    """ 
    We are going to take real life recipes, and hide one ingredient one by one. 
    Then we are going to see if the model correctly identifies that ingredient as the missing one.

    We are going to collect two quantities in "out": 
        1) the order in which the model suggests adding the hidden ingredient (ing[idx]) to be added to the recipe
        2) the probability that it assigns to this ingredient
    """
    for gt_mask in test_data:
        # shorten the recipe by one (in every possible way)
        positive_indices = np.where(gt_mask>0)[0]

        out_sub = []
        for idx in positive_indices:
            shortened_mask = np.copy(gt_mask)
            shortened_mask[idx] = 0

            rng = jax.random.PRNGKey(42)
            sample_probs = sample_expand_jit(rng, params, shortened_mask, num_samples = 1000) # n x 146
            sample_probs = jnp.mean(sample_probs, axis=0)

            mask = sample_probs < 1.0
            valid_indices = jnp.where(mask, size=mask.sum(), fill_value=-1)[0]
            valid_values = sample_probs[valid_indices]
            sort_order = jnp.argsort(-valid_values)
            sorted_values = valid_values[sort_order]
            sorted_indices = valid_indices[sort_order]

            # find the position of the hidden index and see if it is first, also see if its probability is greater than 0.5
            matches = jnp.where(sorted_indices == idx)[0]
            position = int(matches[0]) if matches.size > 0 else -1
            value = float(sorted_values[position])
            out_sub.append([position, value])
        out.append(out_sub)

    with open('params/recipe_expansion_results.pkl', 'wb') as f:
        pickle.dump(out, f)