from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numpy as np
import pickle

def boxcox_transform_forward(recipes):
    data_normalized = []
    fitted_lambdas = []
    means_stds = []
    for i in range(recipes.shape[1]):
        transformed, fitted_lambda = boxcox(recipes[:,i] + 1e-8)
        mean, std = transformed.mean(), transformed.std()
        transformed = (transformed - mean)/std
        data_normalized.append(transformed)
        fitted_lambdas.append(fitted_lambda)
        means_stds.append([mean, std])
    data_normalized = np.array(data_normalized).T

    return data_normalized, fitted_lambdas, means_stds

def boxcox_transform_back(transformed_recipe, fitted_lambdas, means_stds):
    recipe = []
    for i in range(len(transformed_recipe)):
        fitted_lambda = fitted_lambdas[i]
        mean, std = means_stds[i]
        temp = transformed_recipe[i]*std + mean
        temp = inv_boxcox(temp, fitted_lambda)
        recipe.append(temp)
    return np.array(recipe)

# conversions.org
conversions = {
    'kg':       ['gr', 1000],
    'lb':       ['gr', 453.592],
    'oz':       ['gr', 28.3495],
    'tbsp':     ['ml', 14.78676],
    'tsp':      ['ml', 4.92892],
    'fl oz':    ['ml', 29.5735],
    'cup':      ['ml', 236.588],
    'dash':     ['ml', 0.625],
    'gal':      ['ml', 3785.41],
    'qt':       ['ml', 946.353],
    'pt':       ['ml', 473.176],
    'pinch':    ['ml', 0.61611],
    'l':        ['ml', 1000],
    'drop':     ['ml', 0.08214],
}



# https://www.fao.org/4/ap815e/ap815e.pdf , aqua-calc.com, kg-m3.com
"""
Group the following together (i.e. they use the same density):
    1. Dried leaves and flakes
    2. Spices
    3. Cheeses
    4. Mushrooms
    5. Leafy greens
    6. Thin liquids (water, soy sauce, juice etc)
    7. Thick liquids (salad dressing, hot sauce, salsa, etc)
    8. Bulky vegetables (carrots, zucchini, beet, etc)

Assume all vegetables are chopped.
"""
densities = { 
    'salt':     1.28,
    'pepper':   0.58, # Used entry for 'spice, blend'
    'butter':   0.911,
    'mustard':  1.1, # 'salad dressing'
    'lemon': 1,
    'worcestershire sauce': 1.1, # 'salad dressing'
    'taco seasoning': 0.58, # 'spice, blend'
    'chili': 0.5,
    'onion': 0.22, # 'chopped'
    'barbecue sauce': 1.1, # 'salad dressing'
    'olive': 0.65,
    'garlic': 0.13, # 'onions, minced'
    'cinnamon': 0.56,
    'nutmeg': 0.58, # 'spice, blend'
    'sour cream': 1.005,
    'cucumber': 0.54, # 'carrots, raw, chopped'
    'feta': 1.11, # https://www.journalofdairyscience.org/article/S0022-0302(80)82965-1/pdf
    'hot sauce': 1.1, # 'salad dressing'
    'jalapeno': 0.51, # Sweet pepper, raw, cubes
    'water': 1, 
    'vinegar': 1,
    'oil': 0.918, # Oil, vegetable, olive
    'celery': 0.22, # onions, chopped
    'tomato': 0.22, # onions, chopped
    'mozzarella': 0.47, # aqua-calc.com
    'bacon': 0.47, # bacon pieces
    'ginger': 0.41,
    'breadcrumb': 0.45,
    'milk': 1.04,
    'ketchup': 1.15,
    'horseradish': 1.01,
    'chili flake': 0.35, # garlic flakes
    'cheddar cheese': 1.078, #https://eurekamag.com/research/001/061/001061121.php
    'cayenne pepper': 0.58, # 'spice, blend'
    'mushroom': 0.3,
    'mayonnaise': 1.1, # 'salad dressing'
    'coriander seed': 0.58, # 'spice, blend'
    'bean': 0.73,
    'basil': 0.35, # garlic flakes
    'cumin': 0.58, # 'spice, blend'
    'mint': 0.35, # garlic flakes
    'thyme': 0.35, # garlic flakes
    'flour': 0.48,
    'parmesan': 0.63,
    'other seasoning': 0.58, # 'spice, blend'
    'dressing': 1.1, # 'salad dressing'
    'relish': 1.04,
    'sugar': 0.7, # sugar, granulated
    'cabbage': 0.3,
    'italian seasoning': 0.58, # 'spice, blend'
    'oat': 0.43, #kg-m3.com
    'paprika': 0.58, # 'spice, blend'
    'lettuce': 0.3, #chopped cabbage
    'peanut': 0.69,
    'yogurt': 1.06,
    'monterey jack': 1.078, # same as cheeddar cheese
    'blue cheese': 1.078,
    'cream cheese': 1.005, #sour cream
    'chili sauce': 1.1, # 'salad dressing'
    'portobello': 0.3, # same as mushroom
    'gouda': 1.078,
    'cornflour': 0.48, # same as flour
    'soy sauce': 1,
    'broth': 1,
    'sauce': 1.1, # 'salad dressing'
    'lentil': 0.89,
    'sesame seed': 0.61,
    'greens': 0.3, #chopped cabbage
    'carrot': 0.54,
    'chive': 0.3, #chopped cabbage
    'rosemary': 0.3,
    'provolone': 1.078, # same as cheeddar cheese
    'salsa': 1.1, # 'salad dressing'
    'walnut': 0.49,
    'steak seasoning': 0.58, # 'spice, blend'
    'green onion': 0.3, #chopped cabbage
    'honey': 1.415,
    'tofu': 1.07, # raw, firm
    'ranch sauce': 1.1, # 'salad dressing'
    'sage': 0.35, # garlic flakes
    'guacamole': 1.1, # 'salad dressing'
    'teriyaki sauce': 1.1,# 'salad dressing'
    'shallot': 0.22, # 'onions, chopped'
    'corn': 0.72, # corn, shelled
    'pimento': 0.3,
    'cilantro': 0.3,
    'parsley': 0.3,
    'oregano': 0.3,
    'zucchini': 0.54, # carrot
    'pasta': 0.39,
    'goat cheese': 1.078,
    'lime': 1,
    'cranberry': 1,
    'yeast': 0.95,
    'wheat germ': 0.48, # same as flour
    'beet': 0.54, # carrot
    'orange': 1,
    'mango': 1,
    'chipotle chili': 0.51, # Sweet pepper, raw, cubes
    'chickpea': 0.73, # same as bean and corn
    'cream': 1.005, #sour cream
    'curry': 1.1, # 'salad dressing'
    'arugula': 0.3,
    'quinoa': 0.72,
    'beef': 0.96, # ground beef
    'turkey': 0.98,
    'pork': 0.98,
    'chicken': 0.98,
    'egg': 1.1, # https://homebrewersassociation.org/beyond-beer/egg-mead-ium-measuring-gravity-egg/
    'pickle': 0.54, #same as cucumber
    'potato': 0.54,
    'apple': 0.54,
    'pineapple': 0.54,
    'dill': 0.3,
    'spinach': 0.3,
    'avocado': 0.54,
    'other cheese': 1.078,
    'chip': 0.12,
    'cracker': 0.12,
    'noodle': 0.12, # Use the same as chip and cracker for now.
    'bouillon': 1.01,
    'wine': 1,
    'brown sugar': 0.7,
    'hoisin sauce': 1.08,
    'tarragon': 0.33,
    'bulgur': 0.6,
    'rice': 0.83,
    'cottage cheese': 226/236.6,
    'curry powder': 2.0/4.93,
    'sunflower seed': 140/236.6,
    'almond': 141/236.6,
    'beer': 1,
    'liquid smoke': 0.58,
    'chicken seasoning': 0.58,
    'steak sauce': 34/14.79,
    'marjoram': 1.7/4.93,
    'cajun seasoning': 0.58,
    'allspice': 1.9/4.93,
    'pesto': 62/236.6,
    'tahini': 15/14.79,
    'wasabi': 130/236.6,
}

usda_codes = {
    'allspice': 171315,
    'almond': 2707486,
    'apple': 2709215,
    'arugula': 2709791,
    'avocado': 2709223,
    'bacon': 168277,
    'barbecue sauce': 174523,
    'basil': 172232,
    'bean': 173744,
    'beef': 2514743,
    'beer': 168749,
    'beet': 2685576,
    'blue cheese': 172175,
    'bouillon': 171562,
    'breadcrumb': 1930877,
    'broth': 172883,
    'brown sugar': 2710260,
    'buffalo': 175299,
    'bulgur': 2710820,
    'bun': 2707784,
    'butter': 173430,
    'cabbage': 169975,
    'cajun seasoning': 1887349,
    'carrot': 2709660,
    'cayenne pepper': 170932,
    'celery': 2709778,
    'cheddar cheese': 2705709,
    'other cheese': 2705764,
    'chicken': 2706091,
    'chicken seasoning': 2060346,
    'chickpea': 173756,
    'chili': 170497,
    'chili flake': 579086,
    'chip': 2709434,
    'chipotle chili': 2009291,
    'chive': 2709781,
    'cilantro': 169997,
    'cinnamon': 171320,
    'coriander seed': 170922,
    'corn': 2709783,
    'cornflour': 2710835,
    'cottage cheese': 172182,
    'cracker': 2708184,
    'cranberry': 2709279,
    'cream': 2705597,
    'cream cheese': 173418,
    'cucumber': 2709784,
    'cumin': 170923,
    'curry': 2710178,
    'curry powder': 170924,
    'dill': 172233,
    'dough': 172791,
    'dressing': 173592,
    'egg': 171287,
    'feta': 173420,
    'fish': 2706284,
    'flour': 169761,
    'garlic': 169230,
    'ginger': 169231,
    'goat cheese': 2705716,
    'gouda': 171241,
    'green onion': 170005,
    'greens': 2709792,
    'guacamole': 2709307,
    'hoisin sauce': 172886,
    'honey': 169640,
    'horseradish': 173472,
    'hot sauce': 171186,
    'italian seasoning': 1887966,
    'jalapeno': 168576,
    'ketchup': 168556,
    'lamb': 2705907,
    'lemon': 2709168,
    'lentil': 172420,
    'lettuce': 2709789,
    'lime': 168155,
    'liquid smoke': 2680455,
    'mango': 2709242,
    'marjoram': 170928,
    'mayonnaise': 168112,
    'milk': 2705385,
    'mint': 173474,
    'monterey jack': 2705720,
    'mozzarella': 170845,
    'mushroom': 169251,
    'mustard': 171043,
    'noodle': 2708354,
    'nutmeg': 171326,
    'oat': 2708489,
    'oil': 748278,
    'olive': 2710089,
    'onion': 170000,
    'orange': 169918,
    'oregano': 171328,
    'other seasoning': 171331,
    'paprika': 171329,
    'parmesan': 171247,
    'parsley': 170416,
    'pasta': 168927,
    'patty': 174031,
    'peanut': 2515376,
    'pepper': 170931,
    'pesto': 171582,
    'pickle': 169379,
    'pimento': 168559,
    'pineapple': 169124,
    'pork': 2705863,
    'portobello': 169255,
    'potato': 170027,
    'provolone': 170850,
    'quinoa': 168874,
    'ranch sauce': 173592,
    'relish': 168561,
    'rice': 168879,
    'rosemary': 173473,
    'sage': 170935,
    'salsa': 174524,
    'salt': 173468,
    'sauce': 174524,
    'sausage': 174584,
    'sesame seed': 170150,
    'shallot': 170499,
    'sour cream': 2346387,
    'soy sauce': 174278,
    'spinach': 2709614,
    'steak sauce': 171825,
    'steak seasoning': 2157193,
    'sugar': 169655,
    'sunflower seed': 170562,
    'taco seasoning': 172243,
    'tahini': 2707587,
    'tarragon': 170937,
    'teriyaki sauce': 171167,
    'thyme': 173470,
    'tofu': 172476,
    'tomato': 170457,
    'tortilla': 175036,
    'turkey': 171505,
    'vinegar': 172241,
    'walnut': 170187,
    'wasabi': 168583,
    'water': 173647,
    'wheat germ': 168892,
    'wine': 173185,
    'worcestershire sauce': 171610,
    'yeast': 167717,
    'yogurt': 171284,
    'zucchini': 168565,
}

# The nutrients in FDA list (https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels)
fda_list = {
    'added sugars': [50, 'g'],
    'biotin': [30, 'mcg'],
    'calcium': [1300, 'mg'],
    'chloride': [2300, 'mg'],
    'choline': [550, 'mg'],
    'cholesterol': [300, 'mg'],
    'chromium': [35, 'mcg'],
    'copper': [0.9, 'mg'],
    'fiber': [28, 'g'],
    'fat': [78, 'g'],
    'folate': [400, 'mcg'],
    'iodine': [150, 'mcg'],
    'iron': [18, 'mg'],
    'magnesium': [420, 'mg'],
    'manganese': [2.3, 'mg'],
    'molybdenum': [45, 'mcg'],
    'niacin': [16, 'mg'],
    'pantothenic acid': [5, 'mg'],
    'phosphorus': [1250, 'mg'],
    'potassium': [4700, 'mg'],
    'protein': [50, 'g'],
    'riboflavin': [1.3, 'mg'],
    'saturated': [20, 'g'], #saturated fat
    'selenium': [55, 'mcg'],
    'sodium': [2300, 'mg'],
    'thiamin': [1.2, 'mg'],
    'carbohydrate': [275, 'g'],
    'vitamin a': [900, 'mcg'],
    'vitamin b6': [1.7, 'mg'],
    'vitamin b12': [2.4, 'mcg'],
    'vitamin c': [90, 'mg'],
    'vitamin d': [20, 'mcg'],
    'vitamin e': [15, 'mg'],
    'vitamin k': [120, 'mcg'],
    'zinc': [11, 'mg']
}


"""
NASEM LIST 

Use Recommended Dietary Allowances (RDA) (when available) or Adequate Intakes (AI) from the Food and Nutrition Board of the National Academies of Sciences Engineering, and Medicine instead.

Males 19-30 and Males 31-50 groups have nearly identical values, with the exception of magnesium. (400 mg for 19-30, 420 mg for 31-50. Use the average.)

Thus, our values are for Males 19-50.

Sources:
https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx

minerals: https://www.ncbi.nlm.nih.gov/books/NBK545442/table/appJ_tab3/?report=objectonly
vitamins: https://www.ncbi.nlm.nih.gov/books/NBK56068/table/summarytables.t2/?report=objectonly
total water and macronutrients: https://www.ncbi.nlm.nih.gov/books/NBK56068/table/summarytables.t4/?report=objectonly


USDA's food central database does not list the following nutrients in foods:
1. molybdenum (source: https://ods.od.nih.gov/factsheets/Molybdenum-HealthProfessional/)
2. chromium (I couldn't find a source for this, but USDA doesn't list chromium for a lot of foods that are rich in chromium such as potatoes and bananas.)
3. chloride (I couldn't find a source for this, but USDA doesn't list chloride for a lot of foods that are rich in chloride such as salt, tomatoes, olives, celery, lettuce, potatoes, etc.)
Remove these from consideration for the time being.
"""
nasem_list = {
    'calcium': [1000, 'mg', 'ai'],
    # 'chromium': [35, 'mcg', 'ai'], # USDA doesn't list chromium, so remove it from consideration for the time being.
    'copper': [900, 'mcg', 'ai'],
    'fluoride': [4, 'mg', 'ai'],
    'iodine': [150, 'mcg', 'ai'],
    'iron': [8, 'mg', 'ai'],
    'magnesium': [410, 'mg', 'ai'],
    'manganese': [2.3, 'mg', 'ai'],
    # 'molybdenum': [45, 'mcg', 'ai'],  # USDA doesn't list molybdenum, so remove it from consideration for the time being.
    'phosphorus': [700, 'mg', 'ai'],
    'selenium': [55, 'mcg', 'ai'],
    'zinc': [11, 'mg', 'ai'],
    'potassium': [3400, 'mg', 'ai'],
    'sodium': [1500, 'mg', 'ai'],
    # 'chloride': [2.3, 'g', 'ai'],  # USDA doesn't list chloride, so remove it from consideration for the time being.
    'vitamin a': [900, 'mcg', 'rda'],
    'vitamin c': [90, 'mg', 'rda'],
    'vitamin d': [15, 'mcg', 'rda'],
    'vitamin e': [15, 'mg', 'rda'],
    'vitamin k': [120, 'mcg', 'ai'],
    'thiamin': [1.2, 'mg', 'rda'],
    'riboflavin': [1.3, 'mg', 'rda'],
    'niacin': [16, 'mg', 'rda'],
    'vitamin b6': [1.3, 'mg', 'rda'],
    'folate': [400, 'mcg', 'rda'],
    'vitamin b12': [2.4, 'mcg', 'rda'],
    'pantothenic acid': [5, 'mg', 'ai'],
    'biotin': [30, 'mcg', 'ai'],
    'choline': [550, 'mg', 'ai'],
    'carbohydrate': [130, 'g', 'rda'],
    'fiber': [38, 'g', 'ai'],
    'pufa 18:2': [17, 'g', 'ai'], #linoleic acid. Replaced with its omega/"n-" notation because that is what USDA uses
    'pufa 18:3': [1.6, 'g', 'ai'], #α-linoleic acid. Replaced with its omega/"n-" notation because that is what USDA uses
    'protein': [56, 'g', 'rda']
}


# https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fped-overview/
fped_codes = {
    'almond': 42101000,
    'apple': 63101000,
    'arugula': 75113080,
    'avocado': 89902000,
    'bacon': 89902100,
    'barbecue sauce': 74406010,
    'basil': 75109400,
    'bean': 41100990,
    'beef': 21000110,
    'beer': 93101000,
    'beet': 75102500,
    'blue cheese': 14101010,
    'breadcrumb': 99995000,
    'broth': 28310110,
    'brown sugar': 91102010,
    'buffalo': 74406060,
    'bulgur': 56207110,
    'bun': 51154100,
    'butter': 81100500,
    'cabbage': 75103000,
    'carrot': 73101010,
    'celery': 75109000,
    'cheddar cheese': 14104100,
    'other cheese': 14410110, # American. data consisted mostly of american
    'chicken': 24198720,
    'chickpea': 41301990,
    'chili': 89902050,
    'chip': 71200010,
    'chipotle chili': 89902050,
    'chive': 75109500,
    'cilantro': 75109550,
    'corn': 75109600,
    'cornflour': 75109600, # same as corn, since it is just ground corn
    'cottage cheese': 14200100,
    'cracker': 54001000,
    'cranberry': 63207010,
    'cream': 12130100,
    'cream cheese': 14301010,
    'cucumber': 75111000,
    'curry': 81312100,
    'dressing': 83100200,
    'egg': 31101010,
    'feta': 14104400,
    'fish': 26100100,
    'flour': 51154100, # use the same as bun as a compromise.
    'garlic': 75111500,
    'ginger': 75503085, # didn't find raw, so used pickled
    'goat cheese': 14104700,
    'gouda': 14105010,
    'green onion': 75117010,
    'greens': 72118220,
    'guacamole': 63409010,
    'hoisin sauce': 41420250,
    'honey': 91302010,
    'horseradish': 75503090,
    'hot sauce': 75511010,
    'jalapeno': 89902050,
    'ketchup': 74401010,
    'lamb': 23000100,
    'lemon': 61113010,
    'lentil': 41305000,
    'lettuce': 72116000,
    'lime': 61116010,
    'mango': 63129010,
    'mayonnaise': 83107000,
    'milk': 11100000,
    'monterey jack': 14106200,
    'mozzarella': 14107010,
    'mushroom': 75115000,
    'mustard': 75506010,
    'noodle': 56112000, # had to use cooked noodles
    'oat': 57602100,
    'oil': 82105500, # Had to specify a type, so chose canola
    'olive': 75510000,
    'onion': 75117020,
    'orange': 61119010,
    'parmesan': 14108010,
    'parsley': 75119000,
    'pasta': 56130000,
    'peanut': 42111000,
    'pesto': 81302070,
    'pickle': 75503010,
    'pimento': 89902050,
    'pineapple': 63141010,
    'pork': 22002000,
    'portobello': 75115000, # mushrooms
    'potato': 71000100,
    'provolone': 14108400,
    'quinoa': 56204005,
    'ranch sauce': 11440040,
    'relish': 75503020,
    'rice': 56205000, # had to use cooked rice
    'salsa': 74402100,
    'sausage': 25221400,
    'sesame seed': 43103000,
    'shallot': 75117020, # used onions
    'sour cream': 12310100,
    'soy sauce': 41420300,
    'spinach': 72125100,
    'steak sauce': 74406100,
    'sugar': 91101000,
    'sunflower seed': 43102000,
    'tahini': 43103300,
    'teriyaki sauce': 41420400,
    'tomato': 74101000,
    'tortilla': 52215000,
    'turkey': 24201000,
    'vinegar': 64401000,
    'walnut': 42116000,
    'wasabi': 75534550,
    'wheat germ': 57412000,
    'wine': 92801000,
    'worcestershire sauce': 41420450,
    'yeast': 75236000,
    'yogurt': 11400000,
    'zucchini': 75535000, # had to choose "pickled"
    'bouillon': 28310110,
}

sampling_batch_size = 10000

def global_2_batch(idx):
    batch_idx = int(idx // sampling_batch_size)
    local_idx = int(idx % sampling_batch_size)
    return batch_idx, local_idx
def find_recipe_in_batch(idx_recipe, seed=42):
    batch_idx, local_idx = global_2_batch(idx_recipe)
    with open(f"params/e2e_samples_rng_{seed}_batch_{batch_idx}.npy", "rb") as f:
        arr = pickle.load(f)
    recipe = arr[local_idx]
    return recipe

bar_color_dict = {
    'onion': np.array([255, 250, 210])/256,
    'beef': np.array([180, 30, 30])/256,
    'cheese': np.array([255, 200, 50])/256,
    'cheddar cheese': np.array([255, 200, 50])/256,
    'other cheese': np.array([255, 200, 50])/256,
    'bun': np.array([210, 160, 120])/256,
    'breadcrumb': np.array([210, 160, 120])/256,
    'tortilla': np.array([210, 160, 120])/256,
    'pickle': np.array([131, 161, 105])/256,
    'mushroom': np.array([186, 158, 136])/256,
    'portobello': np.array([186, 158, 136])/256,
    'greens': np.array([66, 96, 60])/256,
    'lettuce': np.array([66, 96, 60])/256,
    'jalopeno': np.array([66, 96, 60])/256,
    'turkey': np.array([195, 138, 138])/256,
    'pineapple': np.array([254, 234, 99])/256,
    'teriyaki sauce': np.array([126, 92, 73])/256,
    'fish': np.array([250, 128, 114])/256,
    'bean': np.array([247, 93, 89])/256,
    'corn': 'yellow',
    'oil': np.array([255, 200, 50])/256,
    'chili': 'red',
    'salsa': 'red',
}

def plot_recipe(recipe, name=None):
    import matplotlib.pyplot as plt
    with open('data/burgers_step8.pkl', 'rb') as f:
        recipes, ingr_names, calorie_database = pickle.load(f)
    # Normalize to 500 calories
    calories = 0
    for ingr, qty in zip(ingr_names, recipe):
        calories_per_gram = calorie_database[ingr]
        calories += qty * calories_per_gram
    recipe = recipe/calories*500

    # Only retain the nonzero ingredients
    mask1 = recipe > 0
    recipe_masked, ingr_names_masked = recipe[mask1], ingr_names[mask1]
    mask2 = ingr_names_masked != 'salt' # remove salt from the recipes
    recipe_masked, ingr_names_masked = recipe_masked[mask2], ingr_names_masked[mask2]

    # Sort the ingredients
    mask3 = np.argsort(recipe_masked)[::-1]
    recipe_masked, ingr_names_masked = recipe_masked[mask3], ingr_names_masked[mask3]

    # get the colors
    colors = []
    for ingr_name in ingr_names_masked:
        if ingr_name in bar_color_dict:
            colors.append(bar_color_dict[ingr_name])
        else:
            colors.append('white')

    fig, ax = plt.subplots(figsize=(len(recipe_masked)*0.4,2))
    bars = ax.bar(ingr_names_masked, recipe_masked, edgecolor='black', color=colors)
    values = [bar.get_height() for bar in bars]
    labels = [str(int(np.ceil(v))) for v in values]
    ax.bar_label(bars, labels=labels)
    
    ax.set(ylabel='weight [g]', title=name, yticks=[])
    ax.tick_params(axis='x', labelrotation=90)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, ax