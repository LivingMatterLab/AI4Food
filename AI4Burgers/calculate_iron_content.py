import numpy as np
import pandas as pd
import pickle
from calculate_sds import find_recipe_repetitions
sampling_batch_size = 10000

with open('data/burgers_step8.pkl', 'rb') as f:
    _, ingr_names, calorie_database = pickle.load(f)
n_ingr = len(ingr_names)

with open('data/ingr_iron_contents.pkl', 'rb') as f:
    ingr_iron_content = pickle.load(f)


# Calculate the iron content for all the recipes sampled with this rng_seed
seed = 42
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
        recipe_iron_content = 0
        for i in range(n_ingr):
            quantity = recipe[i]
            recipe_iron_content += quantity * ingr_iron_content[ingr_names[i]]
        out.append(recipe_iron_content)

        # Does it include buns?
        includes_buns.append(recipe[idx_buns])



if __name__ == '__main__':
    out = np.array(out).squeeze()
    includes_buns = np.array(includes_buns).squeeze()
    with open(f'params/all_iron_content_rng_{seed}.pkl', 'wb') as f:
        pickle.dump([out, includes_buns], f)
    
    # Filter recipes with iron content between 15 mg and 20 mg, and calculate the sds wrt all the rest
    indices = np.where((out >= 15) & (out < 20) & (includes_buns > 0))[0]
    if len(indices) == 0:
        print("No recipes satisfying this criteria were found")
    else:
        find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_iron_content_15-20_rng_{seed}.txt')

    # Filter recipes with iron content between 20 mg and 25 mg, and calculate the sds wrt all the rest
    indices = np.where((out >= 20) & (out < 25) & (includes_buns > 0))[0]
    if len(indices) == 0:
        print("No recipes satisfying this criteria were found")
    else:
        find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_iron_content_20-25_rng_{seed}.txt')

    # # Filter recipes with iron content between 25 mg and 30 mg, and calculate the sds wrt all the rest
    # indices = np.where((out >= 25) & (out < 30) & (includes_buns > 0))[0]
    # if len(indices) == 0:
    #     print("No recipes satisfying this criteria were found")
    # else:
    #     find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_iron_content_25-30_rng_{seed}.txt')
    
    # # Filter recipes with iron content between 30 mg and 35 mg, and calculate the sds wrt all the rest
    # indices = np.where((out >= 30) & (out < 35) & (includes_buns > 0))[0]
    # if len(indices) == 0:
    #     print("No recipes satisfying this criteria were found")
    # else:
    #     find_recipe_repetitions(seed, indices, output_fname=f'repeat_counts_iron_content_30-35_rng_{seed}.txt')

