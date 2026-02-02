import numpy as np
import pickle
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import jax.numpy as jnp
from train_diffusion import *
import jax.random as random
rng = random.PRNGKey(2025)
from train_bitflip_diff import *
from train_diffusion_val import *
from train_bitflip_diff import sample as sample_masks
def f_sds_masked(r1, r2): # mask salt and pepper because the recipes I collected don't give the quantities for these two
    out = 0.0
    for i in range(n_ingr):
        if ingr_names[i] == 'salt' or ingr_names[i] == 'pepper': #if salt or pepper, ignore 
            out+= 0
        elif r1[i] + r2[i] == 0:
            out+= 0
        elif r1[i] + r2[i] > 0 and r1[i] * r2[i] == 0:
            out+= 1
        elif max(r1[i], r2[i])/min(r1[i], r2[i]) >= 2:
            out+= 1
    return out


with open('data/burgers_step8.pkl', 'rb') as f:
    recipes, ingr_names, calorie_database = pickle.load(f)
n_recipes = len(recipes)
n_ingr = len(ingr_names)
gt_mask = jnp.array(recipes>0, dtype='int32')
with open(f'params/bitflip_mask_params.npy', 'rb') as f:
      mask_params, _ = pickle.load(f)
with open(f'params/diff_value_params.npy', 'rb') as f:
    val_params, fitted_lambdas, means_stds, train_data, val_data = pickle.load(f)
score_model = ApproximateScore(n_hidden=n_hidden)
trained_score = lambda x, t, mask: score_model.apply(val_params, x, t, mask)


#compare against the big mac recipe too, while you are sampling
with open('data/bm_recipe.npy', 'rb') as f:
    r_bm = pickle.load(f)

seed = 0
rng = random.PRNGKey(seed)

sample_batch_size = 10000
n_batches = 10000
for i in range(n_batches):
    print("Sampling ", i)
    _, rng = jax.random.split(rng)

    mask_samples = sample_masks(rng, mask_params, num_samples=sample_batch_size)
    recipe_samples = reverse_sde(rng, n_ingr, drift, dispersion, trained_score, mask_samples).block_until_ready()

    unnormalized = []
    for j in range(n_ingr):
        ingr, fitted_lambda, (mean, std) = recipe_samples[:,j], fitted_lambdas[j], means_stds[j]
        ingr = ingr*std + mean
        ingr = inv_boxcox(ingr, fitted_lambda)
        unnormalized.append(ingr)

    recipe_samples = np.array(unnormalized).T * np.array(mask_samples)
    recipe_samples[np.isnan(recipe_samples)] = 0

    for j in range(len(recipe_samples)):
        sds = f_sds_masked(recipe_samples[j], r_bm)
        if sds == 0:
            print(i*sample_batch_size+j)
            print(recipe_samples[j])

    with open(f'params/e2e_samples_rng_{seed}_batch_{i}.npy', 'wb') as f:
        pickle.dump(np.array(recipe_samples), f)