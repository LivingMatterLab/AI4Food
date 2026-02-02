import numpy as np
import pandas as pd
import pickle
from calculate_sds import find_recipe_repetitions
sampling_batch_size = 10000

with open('data/burgers_step8.pkl', 'rb') as f:
    _, ingr_names, calorie_database = pickle.load(f)
n_ingr = len(ingr_names)

with open('data/environmental/ingr_env_impact_scores.pkl', 'rb') as f:
    env_score_list = pickle.load(f)


seed = 42
# Calculate the environmental impact scores for all the recipes sampled with this rng_seed
out = []
# Also keep track of whether it includes buns.
idx_buns = np.where(ingr_names=='bun')
includes_buns = []

for batch_idx in range(100):
    fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
    with open(fname, "rb") as f:
        arr = pickle.load(f) #10000 x 146
    for recipe in arr:
        # Normalize the recipe by calories
        recipe_calories = 0
        for i in range(n_ingr):
            name = ingr_names[i]
            ingr_cal_per_gram = calorie_database[name]
            recipe_calories += ingr_cal_per_gram * recipe[i]
        recipe = recipe/recipe_calories*1000

        # Now calculate the environmental impact score
        recipe_impact = 0
        for i in range(n_ingr):
            quantity = recipe[i]
            ingr_impact = env_score_list[i]
            recipe_impact += quantity * ingr_impact
        out.append(recipe_impact)

        # Does it include buns?
        includes_buns.append(recipe[idx_buns])
    if batch_idx % 10 == 0:
        print(f'Calculated environmental scores for {batch_idx}/100 batches')
out = np.array(out).squeeze()
includes_buns = np.array(includes_buns).squeeze()
out = np.nan_to_num(out, nan=10)
with open(f'params/all_env_impacts_rng_{seed}.pkl', 'wb') as f:
    pickle.dump([out, includes_buns], f)


if __name__ == '__main__':
    with open(f'params/all_env_impacts_rng_{seed}.pkl', 'rb') as f:
        out, includes_buns = pickle.load(f)
    
    # # Filter recipes with env 0.2 <= score < 0.3, and calculate the sds wrt all the rest
    # indices = np.where((out >= 0.2) & (out < 0.3) & (includes_buns > 0))[0]
    # if len(indices) == 0:
    #     print("No recipes satisfying this criteria were found")
    # else:
    #     find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_env_score_0.2-0.3_rng_{seed}.txt')

    # indices = np.where((out >= 0.1) & (out < 0.2) & (includes_buns > 0))[0]
    # if len(indices) == 0:
    #     print("No recipes satisfying this criteria were found")
    # else:
    #     find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_env_score_0.1-0.2_rng_{seed}.txt')

    # indices = np.where((out >= 0) & (out < 0.1) & (includes_buns > 0))[0]
    # if len(indices) == 0:
    #     print("No recipes satisfying this criteria were found")
    # else:
    #     find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_env_score_0.0-0.1_rng_{seed}.txt')

    indices = np.where((out >= 0.05) & (out < 0.1) & (includes_buns > 0))[0]
    if len(indices) == 0:
        print("No recipes satisfying this criteria were found")
    else:
        find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_env_score_0.05-0.1_rng_{seed}.txt')

    indices = np.where((out >= 0) & (out < 0.05) & (includes_buns > 0))[0]
    if len(indices) == 0:
        print("No recipes satisfying this criteria were found")
    else:
        find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_env_score_0.0-0.05_rng_{seed}.txt')

