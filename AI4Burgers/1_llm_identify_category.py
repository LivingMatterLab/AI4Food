import numpy as np
import pandas as pd
import ollama
import sys
import kagglehub

if len(sys.argv) < 2:
    print("Please provide a food name")
    sys.exit(1)

plurals = {
    'burger': 'burgers',
    'curry': 'curries',
    'omelette': 'omelettes',
    'soup': 'soups',
}

food = sys.argv[1]
food_plural = plurals[food]

path1 = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
data = pd.read_csv(path1 + '/recipes.csv')

out = []

for idx, row in data.iterrows():
    name = row['Name'].lower() if isinstance(row['Name'], str) else ''

    prompt = f'I will give you the name of a food recipe. I want your help to determine if this is a recipe for a {food}. Output "1" if you think this is a {food} recipe and "0" otherwise. All types of {food_plural} count, but make sure that it is a indeed a {food}. Do not output anything else. Here is the recipe name: "{name}"'
    response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
    response = response['message']['content']

    out.append(response)

out = pd.DataFrame({f'is{food}_llm': out})
out.to_csv(f'data/is{food}_llm.csv', index=False)