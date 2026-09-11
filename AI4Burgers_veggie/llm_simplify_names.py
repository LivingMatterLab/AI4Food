import numpy as np
import pandas as pd
import ollama
import pickle

with open('data/burgers_step2.pkl', 'rb') as f:
    ingr_data, burgers = pickle.load(f)

out = []
for i in range(len(ingr_data)):
    ingr_list = []
    for j in range(len(ingr_data[i])):
        ingr = ingr_data[i][j]
        name = ingr[2]
        prompt = f'I need your help in simplifying the name of a food ingredient. I will give you the description of a food ingredient and I want you to \
            boil it down to its CORE. I want you to remove all the descriptive language and just retain the name of the ingredient itself. This is VERY VERY important: \
                just give me the name of the ingredient without any descriptions or comments. The following are some categories of descriptive words and phrases to look out for \
                    (but these are just example categories, remove all other descriptors too). Example categories of descriptive words and phrases to remove: 1) \
                    Brand names, owner names, creator names. 2) Process names such as chopped, ground, fermented, fried, canned, boiled, baked, etc. \
                        3) Purpose (for example "cooking oil" becomes "oil" and "all-purpose flour" becomes "flour"). 4) Original material that the \
                            ingredient was made of (for example "wheat tortilla" becomes "tortilla"). 5) Marketing terms such as "quick", "nutritional", "old fashioned", etc. 6) \
                            Shapes such as round, waffle cut, etc. 7) Anything about fat content such as "90% lean", "1%", "low fat", etc. 8) Variety names \
                            for fruits and vegetables (for example "anjou pears" becomes "pears", "italian parsley" becomes "parsley", etc.).Remember that \
                            these were example categories. I want you to remove all descriptors and give me just the name of the ingredient. DO NOT WRITE ANYTHING \
                            OTHER THAN THE NAME OF THE INGREDIENT. DO NOT WRITE ANY COMMENTS. With that being said, here is the description of the ingredient: "'
            
        prompt = prompt + name + '"'
        response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
        response = response['message']['content']
        ingr_list.append(response)
    out.append(ingr_list)

with open('data/simplified_names.pkl', 'wb') as f:
    pickle.dump(out, f)
