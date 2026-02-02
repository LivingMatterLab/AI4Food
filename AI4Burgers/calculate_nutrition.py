import numpy as np
import pandas as pd
import pickle
import glob
from calculate_sds import find_recipe_repetitions
from utils import fped_codes
sampling_batch_size = 10000
seed = 42

with open('data/burgers_step8.pkl', 'rb') as f:
    _, ingr_names, calorie_database = pickle.load(f)
n_ingr = len(ingr_names)
with open('data/ingredient_nutrient_info.pkl', 'rb') as f:
    nutrient_names, rdvs, ingredient_nutrients = pickle.load(f) # nutrient names, recommended daily values, ingredient nutrients
idx_carb = np.where(np.array(nutrient_names) == 'carbohydrate')[0][0]
idx_protein = np.where(np.array(nutrient_names) == 'protein')[0][0]
idx_fat = np.where(np.array(nutrient_names) == 'fat')[0][0]
idx_fiber = np.where(np.array(nutrient_names) == 'fiber')[0][0]
idx_potassium = np.where(np.array(nutrient_names) == 'potassium')[0][0]
idx_calcium = np.where(np.array(nutrient_names) == 'calcium')[0][0]
idx_magnesium = np.where(np.array(nutrient_names) == 'magnesium')[0][0]
idx_iron = np.where(np.array(nutrient_names) == 'iron')[0][0]
idx_folate = np.where(np.array(nutrient_names) == 'folate')[0][0]
idx_vita = np.where(np.array(nutrient_names) == 'vitamin a')[0][0]
idx_vitd = np.where(np.array(nutrient_names) == 'vitamin d')[0][0]
idx_vite = np.where(np.array(nutrient_names) == 'vitamin e')[0][0]
idx_vitc = np.where(np.array(nutrient_names) == 'vitamin c')[0][0]
idx_sodium = np.where(np.array(nutrient_names) == 'sodium')[0][0]
idx_sugar = np.where(np.array(nutrient_names) == 'total sugars')[0][0]
idx_sfa = np.where(np.array(nutrient_names) == ' saturated')[0][0]

fped = pd.read_excel('data/FPED_1718.xlsx', sheet_name="FPED_1718", index_col='FOODCODE')
with open('data/ingredient_nutrient_info.pkl', 'rb') as f:
        nutrient_names, rdvs, ingredient_nutrients = pickle.load(f)
nasem_data_male = pd.read_excel('data/nutritional/nasem_dri.xlsx', sheet_name='male')
nasem_data_female = pd.read_excel('data/nutritional/nasem_dri.xlsx', sheet_name='female')
amdr_data = pd.read_excel('data/nutritional/amdr.xlsx')

def calculate_RDI_score(seed, n_batch):
    """
    Calculate the euclidean distance between:
    1. Nutrients in the recipe assuming x calories
    2. RDA
    then, optimize x for euclidean distance for each recipe.

    For optimization, we can actually find it analytically, since this is the standard one-dimensional least squares problem. *
    We want to minimize the distance between the two vectors in Euclidean space:
    ‖xA - B‖²
    Expanding:
    ‖xA - B‖² = (xA - B) • (xA - B) = x²(A • A) - 2x(A • B) + (B • B)
    Differentiate this expression wrt x and set to zero:
    2x(A • A) - 2(A • B) = 0
    Solve for x:
    x = (A • B) / (A • A)

    * This is very important: We need to normalize A and B before doing this. Because the rdvs of some nutrients are very small,
    and in the Euclidean difference they will be ignored. 
    """

    # Normalize rdvs and ingredient_nutrients such that rdvs is a vector of ones...
    temp = rdvs[np.newaxis, :]
    ing_nutr_normalized = ingredient_nutrients/temp
    rdvs_normalized = rdvs/rdvs

    out = []
    for batch_idx in range(n_batch):
        fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
        with open(fname, "rb") as f:
            arr = pickle.load(f) #10000 x 146
        for local_idx, recipe in enumerate(arr):
            recipe_nutrients = np.dot(recipe, ing_nutr_normalized)
            optimized_calories = np.dot(recipe_nutrients, rdvs_normalized)/np.dot(recipe_nutrients, recipe_nutrients)
            optimized_recipe_nutrients = optimized_calories*recipe_nutrients
            euclidean_distance = np.linalg.norm(optimized_recipe_nutrients - rdvs_normalized)
            global_idx = batch_idx*sampling_batch_size + local_idx
            out.append([global_idx, euclidean_distance, optimized_calories])
        if batch_idx % 10 == 0:
            print(f'Calculated euclidean distances for {batch_idx}/{n_batch} batches')
    out = np.array(out)
    out = np.nan_to_num(out, nan=1000000)
    return out

# Calculate HEI scores for the training data
def hei_calculator(recipes):
    out = []
    hei_list = []
    for recipe in recipes:
        # Normalize by 1000 calories
        calories = 0
        for ingr, qty in zip(ingr_names, recipe):
            calories_per_gram = calorie_database[ingr]
            calories += qty * calories_per_gram
        recipe = recipe/calories*1000

        f_total = 0 # total fruits
        v_total = 0 # total vegetables
        v_greens_beans = 0 # greens and beans
        g_whole = 0 # whole grains
        d_total = 0 # total dairy
        pf_total = 0 # total protein foods
        pf_seafd_plant = 0 # total seafood and plant proteins
        pufa_mufa = 0 # polyunsaturaed fats and monounsaturated fats
        sfa = 0 # saturated fats
        g_refined = 0 # refined grains
        sodium = 0 # sodium? sounds like sodium
        add_sugars = 0 # added sugars

        for i, qty in enumerate(recipe):
            ingr = ingr_names[i]
            if ingr in fped_codes:
                fped_code = fped_codes[ingr]
                fped_data = fped.loc[fped_code]

                f_total+= fped_data['F_TOTAL (cup eq.)']/100*qty
                v_total+= fped_data['V_TOTAL (cup eq.)']/100*qty
                v_greens_beans+= fped_data['V_DRKGR (cup eq.)']/100*qty
                v_greens_beans+= fped_data['V_LEGUMES (cup eq.)']/100*qty
                g_whole+= fped_data['G_WHOLE (oz. eq.)']/100*qty
                d_total+= fped_data['D_TOTAL (cup eq.)']/100*qty
                pf_total+= fped_data['PF_TOTAL (oz. eq.)']/100*qty
                pf_seafd_plant+= fped_data['PF_SOY (oz. eq.)']/100*qty
                pf_seafd_plant+= fped_data['PF_NUTSDS (oz. eq.)']/100*qty
                pf_seafd_plant+= fped_data['PF_LEGUMES (oz. eq.)']/100*qty
                g_refined+= fped_data['G_REFINED (oz. eq.)']/100*qty
                add_sugars+= fped_data['ADD_SUGARS (tsp. eq.)']/100*qty

            idx_pufa = np.where(np.array(nutrient_names) == 'polyunsaturated')[0][0]
            idx_mufa = np.where(np.array(nutrient_names) == 'monounsaturated')[0][0]
            idx_sfa = np.where(np.array(nutrient_names) == ' saturated')[0][0]
            pufa_mufa+= ingredient_nutrients[i, idx_pufa]*qty
            pufa_mufa+= ingredient_nutrients[i, idx_mufa]*qty
            sfa+= ingredient_nutrients[i, idx_sfa]*qty

            idx_sodium = np.where(np.array(nutrient_names) == 'sodium')[0][0]
            sodium+= ingredient_nutrients[i, idx_sodium]*qty
        out.append([f_total, f_total, v_total, v_greens_beans, g_whole, d_total, pf_total, pf_seafd_plant, pufa_mufa/sfa, g_refined, sodium, add_sugars, sfa])

        # Now calculate HEI scores
        hei = 0
        # total fruits: 0 - 0.8 cup equivalents
        hei+= np.clip((f_total - 0) / (0.8 - 0) * 5, 0, 5) # assigns 0 to 0 or below, 5 to 0.8 or above, and scales linearly in between
        # whole fruits: Assume all fruits in our recipes are whole
        hei+= np.clip((f_total - 0) / (0.4 - 0) * 5, 0, 5)
        # total vegetables: 0 - 1.1 cup equivalents
        hei+= np.clip((v_total - 0) / (1.1 - 0) * 5, 0, 5)
        # greens and beans: 0 - 0.2 cup equivalents
        hei+= np.clip((v_greens_beans - 0) / (0.2 - 0) * 5, 0, 5)
        # whole grains: 0 - 1.5 oz equivalents
        hei+= np.clip((g_whole - 0) / (1.5 - 0) * 10, 0, 10)
        # dairy: 0 - 1.3 cup equivalents
        hei+= np.clip((d_total - 0) / (1.3 - 0) * 10, 0, 10)
        # total protein foods: 0 - 2.5 oz equivalents
        hei+= np.clip((pf_total - 0) / (2.5 - 0) * 5, 0, 5)
        # seafood and plant proteins: 0 - 0.8 cup equivalents
        hei+= np.clip((pf_seafd_plant - 0) / (0.8 - 0) * 5, 0, 5)
        # (pufa+mufa)/sfa: 1.2 - 2.5
        if sfa == 0:
            hei+= 10
        else:
            hei+= np.clip((pufa_mufa/sfa - 1.2) / (2.5 - 1.2) * 10, 0, 10)
        # In the moderation part below, scores are inversely proportional
        # refined grains: 1.8 - 4.3
        hei+= np.clip((4.3 - g_refined)/(4.3 - 1.8)*10, 0, 10)
        # sodium: 1.1 - 2 g
        hei+= np.clip((2 - sodium)/(2 - 1.1)*10, 0, 10)
        # added sugars: 6.5% to 26% of total energy
        add_sugars_energy = add_sugars*4
        add_sugars_percent = add_sugars_energy/1000*100 # each recipe is 1000 calories.
        hei+= np.clip((26 - add_sugars_percent)/(26 - 6.5)*10, 0, 10)
        # saturated fats: 8% - 16% of total energy
        sfa_energy = sfa*9
        sfa_percent = sfa_energy/1000*100
        hei+= np.clip((16 - sfa_percent)/(16 - 8)*10, 0, 10)

        hei_list.append(hei)
    return np.array(hei_list), np.array(out)

def calculate_bhnds(recipes):
    out = []
    bhnds_list = []
    for recipe in recipes:
        # Normalize by 100 calories. In bhnds it is per 100 kcal
        calories = 0
        for ingr, qty in zip(ingr_names, recipe):
            calories_per_gram = calorie_database[ingr]
            calories += qty * calories_per_gram
        recipe = recipe/calories*100 # in bhnds it is per 100 kcal

        protein = 0
        fiber = 0
        vitd = 0
        potassium = 0
        calcium = 0
        iron = 0
        whole_grain = 0
        vegs = 0
        fruits = 0
        dairy = 0
        nuts_seeds = 0
        sodium = 0
        tot_sugars = 0
        sat_fats = 0

        for i, qty in enumerate(recipe):
            ingr = ingr_names[i]
            if ingr in fped_codes:
                fped_code = fped_codes[ingr]
                fped_data = fped.loc[fped_code]

                whole_grain+= fped_data['G_WHOLE (oz. eq.)']/100*qty
                vegs+= fped_data['V_TOTAL (cup eq.)']/100*qty
                fruits+= fped_data['F_TOTAL (cup eq.)']/100*qty
                dairy+= fped_data['D_TOTAL (cup eq.)']/100*qty
                nuts_seeds+= fped_data['PF_NUTSDS (oz. eq.)']/100*qty

            protein+= ingredient_nutrients[i, idx_protein]*qty
            fiber+= ingredient_nutrients[i, idx_fiber]*qty
            vitd+= ingredient_nutrients[i, idx_vitd]*qty*1_000_000 # convert gram to microgram
            potassium+= ingredient_nutrients[i, idx_potassium]*qty*1_000 # convert gram to miligram
            calcium+= ingredient_nutrients[i, idx_calcium]*qty*1_000 # convert gram to miligram
            iron+= ingredient_nutrients[i, idx_iron]*qty*1_000 # convert gram to miligram
            sodium+= ingredient_nutrients[i, idx_sodium]*qty*1_000 # convert gram to miligram
            tot_sugars+= ingredient_nutrients[i, idx_sugar]*qty
            sat_fats+= ingredient_nutrients[i, idx_sfa]*qty
        out.append([protein, fiber, vitd, potassium, calcium, iron, whole_grain, vegs, fruits, dairy, nuts_seeds, sodium, tot_sugars, sat_fats])

        bhnds = 0
        bhnds+= np.clip(protein/50, 0, 1)
        bhnds+= np.clip(fiber/28, 0, 1)
        bhnds+= np.clip(vitd/20, 0, 1)
        bhnds+= np.clip(potassium/4700, 0, 1)
        bhnds+= np.clip(calcium/1300, 0, 1)
        bhnds+= np.clip(iron/18, 0, 1)

        bhnds+= whole_grain/3
        bhnds+= vegs/2.5
        bhnds+= fruits/2
        bhnds+= dairy/3
        bhnds+= nuts_seeds/0.7

        bhnds-= sodium/2300
        bhnds-= tot_sugars/125
        bhnds-= sat_fats/20

        bhnds*= 100

        bhnds_list.append(bhnds)
    return np.array(bhnds_list), np.array(out)

def piecewise_linear(x, a, b, c, d):
    if x > d:
        return 0
    elif x > c:
        return np.interp(x, [c, d], [1, 0])
    elif x > b:
        return 1
    elif x > a:
        return np.interp(x, [a,b], [0, 1])
    else:
        return 0

def calculate_nna(recipes, gender, age, w, h, pa):
    """
    nna: nestle nutrition algorithm
    w: weight in kg
    h: height in m
    pa: physical activity
    pa: 1.0 (sedentary) - 1.45 (very active) for females
        1.0 (sedentary) - 1.48 (very active) for males
    eer: estimated energy requirement
    """
    score_list = []

    # calculate eer
    if gender == 'male':
        eer = 662 - (9.53*age) + pa*(15.91*w + 539.6*h)
    else:
        eer = 354 - (6.91*age) + pa*(9.36*w + 726*h)


    for temp, recipe in enumerate(recipes):
        calories = 0
        for ingr, qty in zip(ingr_names, recipe):
            calories_per_gram = calorie_database[ingr]
            calories += qty * calories_per_gram
        recipe = recipe/calories*eer

        carb = 0
        protein = 0
        fat = 0
        fiber = 0
        potassium = 0
        calcium = 0
        magnesium = 0
        iron = 0
        folate = 0
        vita = 0
        vitd = 0
        vite = 0
        vitc = 0
        sodium = 0
        sugar = 0
        sfa = 0
        for i, qty in enumerate(recipe):
            ingr = ingr_names[i]
            carb+=      ingredient_nutrients[i, idx_carb]*qty
            protein+=   ingredient_nutrients[i, idx_protein]*qty
            fat+=       ingredient_nutrients[i, idx_fat]*qty
            fiber+=     ingredient_nutrients[i, idx_fiber]*qty
            potassium+= ingredient_nutrients[i, idx_potassium]*qty*1_000 # convert gram to miligram
            calcium+=   ingredient_nutrients[i, idx_calcium]*qty*1_000 # convert gram to miligram
            magnesium+= ingredient_nutrients[i, idx_magnesium]*qty*1_000 # convert gram to miligram
            iron+=      ingredient_nutrients[i, idx_iron]*qty*1_000 # convert gram to miligram
            folate+=    ingredient_nutrients[i, idx_folate]*qty*1_000_000 # convert gram to microgram
            vita+=      ingredient_nutrients[i, idx_vita]*qty*1_000_000 # convert gram to microgram
            vitd+=      ingredient_nutrients[i, idx_vitd]*qty*1_000_000 # convert gram to microgram
            vite+=      ingredient_nutrients[i, idx_vite]*qty*1_000 # convert gram to miligram
            vitc+=      ingredient_nutrients[i, idx_vitc]*qty*1_000 # convert gram to miligram
            sodium+=    ingredient_nutrients[i, idx_sodium]*qty*1_000 # convert gram to miligram
            sugar+=     ingredient_nutrients[i, idx_sugar]*qty
            sfa+=       ingredient_nutrients[i, idx_sfa]*qty

        # now calculate the scores and then take the average
        scores = []

        #------------------------------ amdr-based ----------------------------------------#
        i_group = 0
        upper = amdr_data.iloc[i_group]['age_upper (y)']
        while age > upper:
            i_group += 1
            upper = amdr_data.iloc[i_group]['age_upper (y)']
        amdr_group_data = amdr_data.iloc[i_group]
        for label, cal_per_g, amount in zip([r'carbohydrate (% of energy)', r'protein (% of energy)', r'fat (% of energy)'], 
                                            [4, 4, 9],
                                            [carb, protein, fat]):
            bc = amdr_group_data[label]
            b, c = bc.split('-')
            b, c = float(b), float(c)
            energy = amount*cal_per_g
            energy_percentage = energy/eer*100
            scores.append(piecewise_linear(energy_percentage, 0.5*b, b, c, 1.5*c))


        #------------------------------ dri-based ----------------------------------------#
        i_group = 0
        upper = nasem_data_male.iloc[i_group]['age_upper (y)']
        while age > upper:
            i_group += 1
            upper = nasem_data_male.iloc[i_group]['age_upper (y)'] 
        if gender == 'male':
            nasem_group_data = nasem_data_male.iloc[i_group]
        else:
            nasem_group_data = nasem_data_female.iloc[i_group]
        
        labels = ['total fiber (g/d)', 'potassium (mg/d)', 'calcium (mg/d)', 'magnesium (mg/d)', 'iron (mg/d)',
                    'folate (microgr/d)', 'vitamin a (microgr/d)', 'vitamin d (microgr/d)', 'vitamin e (mg/d)',
                    'vitamin c (mg/d)']
        amounts = [fiber, potassium, calcium, magnesium, iron, folate, vita, vitd, vite, vitc]
        for label, amount in zip(labels, amounts):
            b = nasem_group_data[label]
            if b == 'ND':
                pass
            elif isinstance(b, str):
                b = b.replace('*', '')
                b = float(b)
                scores.append(piecewise_linear(amount, 0.5*b, b, 2*b, 1.5*2*b))
            else:
                scores.append(piecewise_linear(amount, 0.5*b, b, 2*b, 1.5*2*b))
        

        #------------------------------ who-based ----------------------------------------#
        # sodium
        if age > 15:
            c = 2000
            b = 0
            a = -1 # just a dummy negative number to make chart b similar to chart a in fig.1 of the nna paper
            d = 1.5*c
            scores.append(piecewise_linear(sodium, a, b, c, d))
        elif age > 2:
            c = 2000/2000*eer # adjusting based on a 2000 calorie diet for adults
            b = 0
            a = -1
            d = 1.5*c
            scores.append(piecewise_linear(sodium, a, b, c, d))
        # sugar
        sugar_energy = sugar*4
        sugar_energy_percentage = sugar_energy/eer*100
        a = -1
        b = 0
        c = 10
        d = 1.5*c
        scores.append(piecewise_linear(sugar_energy_percentage, a, b, c, d))
        # sfa
        sfa_energy = sfa*9
        sfa_energy_percentage = sfa_energy/eer*100
        a = -1
        b = 0
        c = 10
        d = 1.5*c
        scores.append(piecewise_linear(sfa_energy_percentage, a, b, c, d))




        score_list.append(np.mean(scores))
    return np.array(score_list)




if __name__ == '__main__':
    #------------------------------------ HEI --------------------------------------------
    # all_hei = np.zeros([2, 1_000_000])
    # all_hei[0] = np.arange(1_000_000)
    # seed = 42
    # for batch_idx in range(100):
    #     fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
    #     with open(fname, "rb") as f:
    #         recipes = pickle.load(f) #10000 x 146
    #     hei, _ = hei_calculator(recipes)
    #     all_hei[1, batch_idx*sampling_batch_size : (batch_idx+1)*sampling_batch_size] = hei
    #     print(f'Calculated HEI scores for {batch_idx+1}/100 batches')
    # all_hei = np.nan_to_num(all_hei, nan=0)
    # np.savetxt(f'params/hei_scores_seed_{seed}_1m.txt', all_hei, delimiter=" ")
    # idx_max = np.argmax(all_hei[:,1])
    # print(f'The recipe with the largest HEI score: recipe id: {idx_max}, hei score: {all_hei[:, idx_max]}')
    # threshold_95_percent = np.percentile(all_hei[1,:], 95)
    # best_5_percent = all_hei[:,all_hei[1,:]>threshold_95_percent]
    # np.savetxt(f'params/hei_scores_seed_{seed}_95th_perc.txt', best_5_percent, delimiter=" ")
    # indices = best_5_percent[0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_hei_95th_percentile_rng_{seed}.txt')
    # threshold_90_percent = np.percentile(all_hei[1,:], 90)
    # best_10_percent = all_hei[:,all_hei[1,:]>threshold_90_percent]
    # np.savetxt(f'params/hei_scores_seed_{seed}_90th_perc.txt', best_10_percent, delimiter=" ")
    # indices = best_10_percent[0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_hei_90th_percentile_rng_{seed}.txt')


    # #------------------------------------ bHNDS --------------------------------------------
    # all_bhnds = np.zeros([2, 1_000_000])
    # all_bhnds[0] = np.arange(1_000_000)
    # seed = 42
    # for batch_idx in range(100):
    #     fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
    #     with open(fname, "rb") as f:
    #         recipes = pickle.load(f) #10000 x 146
    #     bhnds, _ = calculate_bhnds(recipes)
    #     all_bhnds[1, batch_idx*sampling_batch_size : (batch_idx+1)*sampling_batch_size] = bhnds
    #     print(f'Calculated bHNDS scores for {batch_idx+1}/100 batches')

    # all_bhnds = np.nan_to_num(all_bhnds, nan=0)
    # np.savetxt(f'params/bhnds_scores_seed_{seed}_1m.txt', all_bhnds, delimiter=" ")
    # idx_max = np.argmax(all_bhnds[1])
    # print(f'The recipe with the largest bHNDS score: recipe id: {idx_max}, bHNDS score: {all_bhnds[:, idx_max]}')
    # threshold_95_percent = np.percentile(all_bhnds[1,:], 95)
    # best_5_percent = all_bhnds[:,all_bhnds[1,:]>threshold_95_percent]
    # np.savetxt(f'params/bhnds_scores_seed_{seed}_95th_perc.txt', best_5_percent, delimiter=" ")
    # indices = best_5_percent[0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_bhnds_95th_percentile_rng_{seed}.txt')
    # threshold_90_percent = np.percentile(all_bhnds[1,:], 90)
    # best_10_percent = all_bhnds[:,all_bhnds[1,:]>threshold_90_percent]
    # np.savetxt(f'params/bhnds_scores_seed_{seed}_90th_perc.txt', best_10_percent, delimiter=" ")
    # indices = best_10_percent[0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_bhnds_90th_percentile_rng_{seed}.txt')


    # #------------------------------------ nna --------------------------------------------
    for gender, age, w, h, pa in [['male', 30, 80, 1.8, 1.0],
                                  ['female', 30, 60, 1.6, 1.0],
                                  ['male', 15, 55, 1.6, 1.4],
                                  ['female', 70, 70, 1.7, 1.2]]:
        all_nna = np.zeros([2, 1_000_000])
        all_nna[0] = np.arange(1_000_000)
        seed = 42
        for batch_idx in range(100):
            fname = f'params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy'
            with open(fname, "rb") as f:
                recipes = pickle.load(f) #10000 x 146
            nna = calculate_nna(recipes, gender, age, w, h, pa)
            all_nna[1, batch_idx*sampling_batch_size : (batch_idx+1)*sampling_batch_size] = nna
            print(f'Calculated NNA scores for {batch_idx+1}/100 batches')

        all_nna = np.nan_to_num(all_nna, nan=0)
        np.savetxt(f'params/nna_scores_{gender}_{age}_{w}_{h}_{pa}_seed_{seed}_1m.txt', all_nna, delimiter=" ")
        idx_max = np.argmax(all_nna[1])
        print(f'The recipe with the largest NNA score: recipe id: {idx_max}, NNA score: {all_nna[:, idx_max]}')
        threshold_95_percent = np.percentile(all_nna[1,:], 95)
        best_5_percent = all_nna[:,all_nna[1,:]>threshold_95_percent]
        np.savetxt(f'params/nna_scores_{gender}_{age}_{w}_{h}_{pa}_seed_{seed}_95th_perc.txt', best_5_percent, delimiter=" ")
        indices = best_5_percent[0]
        find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_nna_{gender}_{age}_{w}_{h}_{pa}_95th_percentile_rng_{seed}.txt')
        threshold_90_percent = np.percentile(all_nna[1,:], 90)
        best_10_percent = all_nna[:,all_nna[1,:]>threshold_90_percent]
        np.savetxt(f'params/nna_scores_{gender}_{age}_{w}_{h}_{pa}_seed_{seed}_90th_perc.txt', best_10_percent, delimiter=" ")
        indices = best_10_percent[0]
        find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_nna_{gender}_{age}_{w}_{h}_{pa}_90th_percentile_rng_{seed}.txt')



    #------------------------- NASEM's RDA and AI values (old) ----------------------------
    # out = calculate_RDI_score(seed=42, n_batch=100)
    # with open(f'params/all_nutrition_eucl_dist_rng_{seed}.pkl', 'wb') as f:
    #     pickle.dump(out, f)
    # threshold_5_percent = np.percentile(out[:,1], 5)
    # best_5_percent = out[out[:,1]<threshold_5_percent]
    # threshold_10_percent = np.percentile(out[:,1], 10)
    # best_10_percent = out[out[:,1]<threshold_10_percent]
    # with open(f'params/nutrition_eucl_dist_95th_perc_rng_{seed}.pkl', 'wb') as f:
    #     pickle.dump(best_5_percent, f)
    # with open(f'params/nutrition_eucl_dist_90th_perc_rng_{seed}.pkl', 'wb') as f:
    #     pickle.dump(best_10_percent, f)
    # # Now look at the most repeated recipes
    # with open(f'params/nutrition_eucl_dist_95th_perc_rng_{seed}.pkl', 'rb') as f:
    #     best_5_percent = pickle.load(f)
    # with open(f'params/nutrition_eucl_dist_90th_perc_rng_{seed}.pkl', 'rb') as f:
    #     best_10_percent = pickle.load(f)
    # indices = best_5_percent[:,0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_nutr_95th_percentile_rng_{seed}.txt')
    # # The best 10% (i.e. 90th percentile in terms of negative euclidean distance)
    # indices = best_10_percent[:,0]
    # find_recipe_repetitions(seed, indices.astype(int), output_fname=f'repeat_counts_nutr_90th_percentile_rng_{seed}.txt')