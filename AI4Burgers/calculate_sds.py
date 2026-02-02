import os
import re
import numpy as np
import pickle
import jax.numpy as jnp
import jax
sampling_batch_size = 10000


with open('data/burgers_step8.pkl', 'rb') as f:
    recipes, ingr_names, calorie_database = pickle.load(f)
n_ingr = len(ingr_names)

def f_sds(r1, r2):
    out = 0.0
    for i in range(n_ingr):
        if r1[i] + r2[i] == 0:
            out+= 0
        elif r1[i] + r2[i] > 0 and r1[i] * r2[i] == 0:
            out+= 1
        elif max(r1[i], r2[i])/min(r1[i], r2[i]) >= 2:
            out+= 1
    return out

# ======================
# SDS FUNCTIONS (JAX)
# ======================

def batched_sds_jax(r_sampled, recipes):
    """
    Compute SDS between a single sampled recipe and all training recipes.
    r_sampled: (n_ingr,)
    recipes: (N_recipes, n_ingr)
    Returns: (N_recipes,) SDS scores
    """
    r1 = jnp.asarray(r_sampled)          # (n_ingr,)
    r2 = jnp.asarray(recipes)            # (N_recipes, n_ingr)

    both_zero = (r1 + r2) == 0
    one_zero = ((r1 + r2) > 0) & ((r1 * r2) == 0)

    ratio_mask = ~(both_zero | one_zero)
    ratio = jnp.where(r1 * r2 != 0,
                      jnp.maximum(r1, r2) / jnp.minimum(r1, r2),
                      0)
    too_large = ratio_mask & (ratio >= 2)

    sds_vals = jnp.sum(one_zero | too_large, axis=1)  # sum over ingredients
    return sds_vals

# JIT compile for speed
batched_sds_jit = jax.jit(batched_sds_jax)


# ======================
# MAIN PIPELINE
# ======================

def process_seed(seed, input_dir="params", output_dir="params"):
    """
    Process all batches for a given RNG seed using JAX and compute the minimum SDS values.
    """
    pattern = re.compile(rf"e2e_samples_rng_{seed}_batch_(\d+)\.npy")
    files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if pattern.match(f)
    ]
    
    if not files:
        raise FileNotFoundError(f"No files found for seed {seed} in {input_dir}")

    # Sort files by batch index
    files.sort(key=lambda x: int(pattern.match(os.path.basename(x)).group(1)))

    all_min_sds = []

    for file in files:
        print(f"Processing {file}...")
        with open(file, "rb") as f:
            r_samples = pickle.load(f)   # shape (20000, n_ingr)

        batch_min_sds = []

        for r_sampled in r_samples:
            sds_vals = batched_sds_jit(r_sampled, recipes)  # vectorized SDS
            min_sds = jnp.min(sds_vals)
            batch_min_sds.append(min_sds)

        # Convert batch result to int8
        batch_min_sds = jnp.array(batch_min_sds, dtype=jnp.int8)
        all_min_sds.append(batch_min_sds)

    # Concatenate results across batches
    all_min_sds = jnp.concatenate(all_min_sds, axis=0).astype(jnp.int8)

    # Save final result
    output_path = os.path.join(output_dir, f"sds_training_data_rng_{seed}.npy")
    with open(output_path, "wb") as f:
        pickle.dump(all_min_sds, f)

    print(f"Saved results to {output_path}")
    return output_path

# seeds = [42]
# for seed in seeds:
#     process_seed(seed)

def obj_sds(sds_file, target_sds = 3):
    with open(sds_file, "rb") as f:
        all_sds = pickle.load(f)
    all_sds = jnp.asarray(all_sds)[:1_000_000]
    indices = jnp.where(all_sds == target_sds)[0]
    n = len(indices)
    print(f"Found {n} recipes with sds == {target_sds}")
    if n == 0:
        raise ValueError("No recipes with the target SDS found.")
    return indices

def obj_inclusion_exclusion(seed, criteria, n_batch=100):
    indices = []
    for batch_idx in range(n_batch): #look at the first 100 batches only (i.e. first 100x10000=1m recipes)
        fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
        with open(fname, "rb") as f:
            arr = pickle.load(f)

        mask = np.ones(len(arr), dtype=bool)
        for name, criterion in criteria:
            idx = np.where(ingr_names == name)
            if criterion == 0:
                mask &= np.squeeze(arr[:, idx] == 0)
            elif criterion == 1:
                mask &= np.squeeze(arr[:, idx] > 0)
        batch_indices = np.squeeze(np.argwhere(mask)).reshape([-1,])
        indices.extend(batch_idx*sampling_batch_size + batch_indices)
    print(f'Found {len(indices)} recipes that match the provided criteria')
    return np.array(indices)

def load_selected_recipes(indices, input_dir, seed):
    # --- Load corresponding recipes
    pattern = re.compile(rf"e2e_samples_rng_{seed}_batch_(\d+)\.npy")
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if pattern.match(f)]
    files.sort(key=lambda x: int(pattern.match(os.path.basename(x)).group(1)))

    # Compute batch sizes for global index mapping
    recipes_selected = []

    i = 0
    idx = indices[i]
    batch_idx = idx // sampling_batch_size
    local_idx = idx % sampling_batch_size
    batch_idx_prev = batch_idx

    while i < len(indices)-1:
        with open(files[batch_idx], "rb") as f:
            arr = pickle.load(f)
        while i < len(indices)-1 and batch_idx == batch_idx_prev:
            recipes_selected.append(arr[local_idx])
            i+= 1
            idx = indices[i]
            batch_idx = idx // sampling_batch_size
            local_idx = idx % sampling_batch_size
        batch_idx_prev = batch_idx


    recipes_selected = jnp.array(recipes_selected)
    print(f"Loaded {recipes_selected.shape[0]} recipes into memory.")
    return recipes_selected


def find_recipe_repetitions(seed, indices, output_fname, block_size=10000, input_dir="params"):
    """
    Compute the most repeated recipe among indices in the samples with rng_seed=seed

    Importantly, this function only looks for repetitions within the list "indices", so be mindful.
    """
    n = len(indices)
    recipes_selected = load_selected_recipes(indices, input_dir, seed)
    # --- Compute how many times sds = 0 wrt the current recipe
    repeat_counts = jnp.zeros(n, dtype=jnp.int32)
    for i in range(n):
        r_i = recipes_selected[i]
        count = 0
        # Compare against all others in blocks
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = recipes_selected[start:end]
            sds_vals = batched_sds_jit(r_i, block)
            count += int(jnp.sum(sds_vals == 0))

        # Subtract self-comparison
        count -= 1
        repeat_counts = repeat_counts.at[i].set(count)

        if i % 100 == 0:
            print(f"Finished finding repeat counts for {i}/{n} recipes")

    # --- Save both counts and indices
    output_path = os.path.join(input_dir, output_fname)
    data = np.column_stack((indices, repeat_counts))
    np.savetxt(output_path, data, fmt="%d", delimiter=" ")

    print(f"Saved zero counts and indices to {output_path}")
    return indices, repeat_counts

def find_repetitions_large(seed, indices, output_fname, block_size=10000, input_dir="params"):
    """
    This function was created to make find_recipe_repetitions scalable to 1m recipes. 
    The only difference is that it saves intermediate steps.
    """
    n = len(indices)
    recipes_selected = load_selected_recipes(indices, input_dir, seed)
    # --- Compute how many times sds = 0 wrt the current recipe
    for batch in range(100):
        repeat_counts = jnp.zeros(sampling_batch_size, dtype=jnp.int32)
        indices_batch = indices[batch*sampling_batch_size : (batch+1)*sampling_batch_size]
        for i in range(sampling_batch_size):
            r_i = recipes_selected[batch*sampling_batch_size + i]
            count = 0
            # Compare against all others in blocks
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                block = recipes_selected[start:end]
                sds_vals = batched_sds_jit(r_i, block)
                count += int(jnp.sum(sds_vals == 0))

            # Subtract self-comparison
            count -= 1
            repeat_counts = repeat_counts.at[i].set(count)

            if i%100 == 0:
                print(f'Processed {i//100}/100 of batch {batch}')
        
        print(f'Finished finding repeat counts for {batch+1}/{100} batches')
        output_path = os.path.join(input_dir, output_fname + f'_batch_{batch}.txt')
        data = np.column_stack((indices_batch, repeat_counts))
        np.savetxt(output_path, data, fmt="%d", delimiter=" ")

if __name__ == '__main__':
    seed = 42
    # target_sds = 3
    # sds_file = f"params/sds_training_data_rng_{seed}.npy"
    # indices = obj_sds(sds_file, target_sds)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_with_indices_sds_{target_sds}_rng_{seed}.txt")

    # target_sds = 4
    # sds_file = f"params/sds_training_data_rng_{seed}.npy"
    # indices = obj_sds(sds_file, target_sds)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_with_indices_sds_{target_sds}_rng_{seed}.txt")

    # target_sds = 5
    # sds_file = f"params/sds_training_data_rng_{seed}.npy"
    # indices = obj_sds(sds_file, target_sds)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_with_indices_sds_{target_sds}_rng_{seed}.txt")

    # target_sds = 6
    # sds_file = f"params/sds_training_data_rng_{seed}.npy"
    # indices = obj_sds(sds_file, target_sds)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_with_indices_sds_{target_sds}_rng_{seed}.txt")

    # # Find recipes that a) have mushrooms, b) don't have meat
    # criteria = [['beef', 0], ['mushroom', 1], ['bun', 1]]
    # indices = obj_inclusion_exclusion(seed, criteria)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_mushroom_nomeat_rng_{seed}.txt")


    # # Find recipes that a) have mushrooms, b) have meat
    # criteria = [['beef', 1], ['mushroom', 1], ['bun', 1]]
    # indices = obj_inclusion_exclusion(seed, criteria)
    # find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_mushroom_meat_rng_{seed}.txt")


    # # find sds for ALL 1 million recipes. Will take time.
    # indices = np.arange(1_000_000)
    # find_repetitions_large(seed, indices, output_fname=f'repeat_counts_first_1m_rng_{seed}')

    # Find the custom recipe that Ellen mentioned, i.e. must contain egg, oatmeal, yogurt, cheese, onion, baking powder. Baking powder is not in the ingredient list.
    criteria = [['egg', 1], ['oat', 1], ['yogurt', 1], ['other cheese', 1], ['onion', 1]]
    indices = obj_inclusion_exclusion(seed, criteria)
    find_recipe_repetitions(seed, indices, output_fname=f"repeat_counts_ellen_custom_rng_{seed}.txt")













