import numpy as np
import pandas as pd
import ollama
import pickle

df = pd.read_csv('data/burgers.csv')
out = []

for idx, row in df.iterrows():
    ing_list = eval(row['ingredients_raw'])
    parsed = []
    for ing in ing_list:
        ing = ing.split(" or ")[0]  # If the recipe offers two alternatives, only keep the first.

        prompt = f'I need your help simplifying ingredients of a food recipe. I will give you one ingredient, you need to tell me 3 things: \
        quantity, units, and name of the ingredient. Use the following comma separated format: "quantity, unit, name". DO NOT WRITE anything else.\
            The quantity has to be ONE real number, so perform the necessary multiplications if necessary. Try to convert everything to units of volume \
                or mass/weight whenever possible. If no quantity is provided write down "NaN" for quantity. If the quantity represents the count of something \
                    write down "-" for "unit". If a range of quantities is given, use the average of the range (e.g. "1-2 tomatoes" becomes "1.5 tomatoes"). \
                        Ignore all extra comments. Ignore descriptions of the ingredients such as "small", "large", "chopped", "fresh", \
                        etc. Ignore all other unnecessary text. I just want you to write down the quantity of the ingredient, unit of measurement, and name \
                            separated by comas. That is all. Example 1: If your input is "1 lb beef, ground" you must type "1, lb, beef". Example 2: If \
                                your input is "1 small onion, finely chopped" you must type "1, -, onion". Example 3: If your input is "1 (4 ounce) can green \
                                    chilies, chopped, drained" you must type "4, oz, chilies". Example 4: If your input is "1 -2 ounce roma tomatoes" \
                                        you must type "1.5, oz, tomatoes" (since 1.5 is the average of the range "1-2".). With that being said, here is your actual input: "'
        prompt = prompt + ing + '"'
        response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
        response = response['message']['content']
        parsed.append(response)
    out.append(parsed)

with open('data/burger_ingredients.pkl', 'wb') as f:
    pickle.dump(out, f)
