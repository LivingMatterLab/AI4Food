import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from functools import partial
import pickle
from train_bitflip_diff import *
from recipe_expansion import posterior_step
rng = jax.random.PRNGKey(42)

# Modify this function such that it only calls 1 step of ancestral sampling (the last step)
def sample_contract(rng, params, observed_mask, num_samples = 1000):
    x = jax.random.bernoulli(rng, 0.5, (num_samples, D)).astype(jnp.int32) # x_T ~ Bern(0.5)
    x = jnp.where(observed_mask == 1, 1, x)
    t = 0 # rather than iterating from T to 0, only do the last step
    probs = posterior_step(rng, params, x, t)
    return probs
sample_contract_jit = jax.jit(sample_contract, static_argnums=3)

if __name__ == "__main__":
    # Load the data and the trained params
    with open(f'params/bitflip_mask_params.npy', 'rb') as f:
        params, _ = pickle.load(f)

    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    n_recipes = len(recipes)
    gt_mask = np.array(recipes>0, dtype='int32')
    split_point = int(0.8*n_recipes) #80-20 split
    test_data = gt_mask[split_point:]

    out = []
    for i, gt_mask in enumerate(test_data):
        print(i)
        out_sub = []
        zero_indices = np.where(gt_mask == 0)[0]

        for idx in zero_indices:
            rng, step_rng = jax.random.split(rng)
            gt_mask[idx] = 1 # temporarily add a new ingredient to the recipe

            probs = sample_contract_jit(step_rng, params, gt_mask, num_samples = 1000)
            probs = jnp.mean(probs, axis=0)

            flipped_value = probs[idx]
            active_probs = probs[gt_mask == 1]

            rank = np.sum(active_probs < flipped_value)

            out_sub.append((rank, flipped_value))

            gt_mask[idx] = 0
        out.append(out_sub)
    
    with open('params/recipe_contraction_results.pkl', 'wb') as f:
        pickle.dump(out, f)
