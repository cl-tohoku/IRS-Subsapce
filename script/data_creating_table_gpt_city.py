from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import random
import json
from tqdm import tqdm
import sys
from multiprocessing import Pool

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer_mistral = AutoTokenizer.from_pretrained(model_id)
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer_llama = AutoTokenizer.from_pretrained(model_id)
model_id = "Qwen/Qwen3-8B"
tokenizer_qwen = AutoTokenizer.from_pretrained(model_id)

from openai import OpenAI
import os
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key=api_key)

def gen_story_via_table(table, description, atts):
    ATT0, ATT1, ATT2, ATT3, ATT4 = atts
    prompt = f"""
    You are given a table of entities with corresponding attributes and a description:
    
    Table:
    {table}
    
    Description:
    {description}

    Write a short, coherent story under 100 words that includes all facts in the table. For each entity:  
    1. Start with their {ATT0}.
    2. Then methion their {ATT1}.   
    3. Then mention their {ATT2}.  
    4. Then mention their {ATT3}.
    5. At last mention their {ATT4}.
    6. Do not directly copy the description into the story.
    7. Use ALL the exact words from the table (do not change singular to plural, spelling, or format).
    8. Do not change the order of facts.
    9. Do not use "*" mark in the story.
    10. Rewrite the description in a coherent way but do not copy the description.
    11. Each fact should be in the order of relation to object, rather than object to relation.
    Strictly keep this order of relations for every entity.  
    Write full sentences and combine all facts naturally into one story.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or "gpt-4.1-mini", "gpt-4o", etc.
        messages=[
            {"role": "system", "content": "You are a helpful writing assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,   # limits story length
        temperature=0.7   # creativity (0 = factual, 1 = creative)
    )
    
    story = response.choices[0].message.content
    return story


def load_elements():
    career = ['teacher', 'doctor', 'scientist', 'chef', 'nurse', 'engineer', 'artist', 'writer', 'actor', 'pilot', 'driver', 'farmer', 'judge', 'lawyer', 'coach', 'student', 'soldier', 'singer', 'dancer', 'builder', 'guard', 'manager', 'clerk', 'baker', 'driver', 'miner', 'fisher', 'plumber', 'carpenter', 'painter', 'programmer', 'designer', 'technician', 'receptionist', 'journalist', 'editor', 'scientist', 'researcher', 'athlete', 'pilot', 'officer', 'coach', 'surgeon', 'waiter', 'waitress', 'driver', 'cashier', 'mechanic', 'fireman', 'police']
    
    city = ['Paris', 'London', 'Berlin', 'Madrid', 'Rome', 'Milan', 'Naples', 'Turin', 'Venice', 'Florence', 'Tokyo', 'Osaka', 'Nagoya', 'Kobe', 'Kyoto', 'Sapporo', 'Sendai', 'Fukuoka', 'Hiroshima', 'Yokohama', 'Seoul', 'Busan', 'Daegu', 'Incheon', 'Daejeon', 'Gwangju', 'Beijing', 'Shanghai', 'Shenzhen', 'Guangzhou', 'Chengdu', 'Wuhan', 'Tianjin', 'Nanjing', 'Suzhou', 'Hangzhou', 'Chongqing', 'Xian', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Pune', 'Jaipur', 'Kanpur', 'Lucknow', 'Nagpur', 'Bhopal', 'Hyderabad', 'Karachi', 'Lahore', 'Islamabad', 'Quetta', 'Peshawar', 'Multan', 'Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Denpasar', 'Yogyakarta', 'Bangkok', 'Chiangmai', 'Phuket', 'Hanoi', 'Danang', 'Haiphong', 'Hue', 'Saigon', 'Cairo', 'Alexandria', 'Giza', 'Luxor', 'Aswan', 'Khartoum', 'Tunis', 'Rabat', 'Casablanca', 'Fez', 'Marrakech', 'Algiers', 'Oran', 'Tripoli', 'Benghazi', 'Accra', 'Kumasi', 'Lagos', 'Abuja', 'Kano', 'Ibadan', 'Kaduna', 'Nairobi', 'Mombasa', 'Kisumu', 'Addis', 'Dakar', 'Bamako', 'Abidjan', 'Monrovia', 'Freetown', 'Luanda', 'Maputo', 'Harare', 'Bulawayo', 'Pretoria', 'Durban', 'Soweto', 'Johannesburg', 'CapeTown', 'Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Darwin', 'Hobart', 'Auckland', 'Wellington', 'Hamilton', 'Christchurch', 'Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Ottawa', 'Edmonton', 'Winnipeg', 'Halifax', 'Quebec', 'Regina', 'Chicago', 'Houston', 'Dallas', 'Austin', 'Phoenix', 'Denver', 'Boston', 'Seattle', 'Miami', 'Orlando', 'Atlanta', 'Detroit', 'Cleveland', 'Pittsburgh', 'Baltimore', 'Nashville', 'Memphis', 'Tampa', 'SanDiego', 'Portland', 'Columbus', 'Cincinnati', 'Indianapolis', 'Milwaukee', 'Minneapolis', 'StLouis', 'Kansas', 'Omaha', 'Dublin', 'Cork', 'Lisbon', 'Porto', 'Warsaw', 'Krakow', 'Wroclaw', 'Gdansk', 'Poznan', 'Prague', 'Brno', 'Vienna', 'Graz', 'Salzburg', 'Innsbruck', 'Budapest', 'Debrecen', 'Athens', 'Thessaloniki', 'Patras', 'Sofia', 'Plovdiv', 'Varna', 'Bucharest', 'Cluj', 'Timisoara', 'Iasi', 'Belgrade', 'NoviSad', 'Skopje', 'Tirana', 'Podgorica', 'Sarajevo', 'Mostar', 'Zagreb', 'Split', 'Ljubljana', 'Stockholm', 'Gothenburg', 'Malmo', 'Oslo', 'Bergen', 'Stavanger', 'Trondheim', 'Copenhagen', 'Aarhus', 'Helsinki', 'Tampere', 'Turku', 'Tallinn', 'Riga', 'Vilnius', 'Minsk', 'Moscow', 'Kazan', 'Samara', 'Sochi', 'Omsk', 'Perm', 'Ufa', 'Tula', 'Rostov', 'Kiev', 'Lviv', 'Odessa', 'Kharkiv']

    obj1 = ['cup', 'pen', 'book', 'chair', 'table', 'lamp', 'plate', 'spoon', 'fork', 'knife', 'bottle', 'glass', 'wallet', 'phone', 'key', 'notebook', 'pencil', 'eraser', 'ruler', 'clock', 'box', 'bag', 'hat', 'shoe', 'sock', 'shirt', 'jacket', 'door', 'window', 'chair', 'candle', 'ball', 'balloon', 'plant', 'flower', 'tree', 'brush', 'comb', 'mirror', 'paper', 'envelope', 'stamp', 'magnet', 'broom', 'bucket', 'towel', 'blanket', 'pillow', 'toy', 'brush', 'camera', 'lampstand', 'bottlecap']
    
    obj2 = ['cup', 'pen', 'bottle', 'chair', 'table', 'plate', 'book', 'bag', 'lamp', 'vase', 'can', 'jar', 'pencil', 'notebook', 'sofa', 'clock', 'shoe', 'hat', 'glass', 'candle', 'cupboard', 'mirror', 'tray', 'bucket', 'ball', 'rug', 'towel', 'wallet', 'key', 'magnet', 'chair', 'bowl', 'spoon', 'fork', 'knife', 'mat', 'blanket', 'pillow', 'toy', 'computer', 'monitor', 'keyboard', 'mouse', 'phone', 'tablet', 'radio', 'camera', 'drum', 'guitar', 'violin', 'canvas', 'glove', 'boot', 'sock', 'ring', 'necklace', 'bracelet', 'wallet', 'belt', 'helmet', 'tub', 'tray', 'pan', 'pot', 'kettle', 'mug', 'bottle', 'cup', 'glass', 'plate', 'jar', 'can', 'basket', 'cabinet', 'locker', 'bucket', 'chair', 'table']

    nation = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Croatia', 'Cuba', 'Cyprus', 'Denmark', 'Djibouti', 'Dominica', 'Ecuador', 'Egypt', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Samoa', 'Senegal', 'Serbia', 'Seychelles', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon', 'Somalia', 'Spain', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']
    name_male1 = ['Alex', 'Ben', 'Bob', 'Carl', 'Dan', 'Dave', 'Ed', 'Eli', 'Eric', 'Finn', 'Fred', 'Gabe', 'Gary', 'Glen', 'Hank', 'Ian', 'Jack', 'Jake', 'James', 'Jay', 'Jeff', 'Jim', 'Joe', 'Joel', 'John', 'Josh', 'Kyle', 'Lee', 'Leo', 'Liam', 'Matt', 'Max', 'Mike', 'Nick', 'Noah', 'Owen', 'Paul', 'Pete', 'Ray', 'Rick', 'Rob', 'Roy', 'Ryan', 'Sam', 'Sean', 'Seth', 'Toby', 'Todd', 'Tom', 'Troy']
    name_male2 = ['John', 'Jack', 'Luke', 'Paul', 'Mark', 'Alan', 'Owen', 'Noah', 'Liam', 'Kyle', 'Evan', 'Eric', 'Ryan', 'Sean', 'Adam', 'Gary', 'Joel', 'Josh', 'Carl', 'Ivan', 'Nick', 'Tony', 'Jose', 'Hugo', 'Levi', 'Troy', 'Brad', 'Dean', 'Greg', 'Zack', 'Jake', 'Matt', 'Pete', 'Alex', 'Ross', 'Finn', 'Drew', 'Chad', 'Sean', 'Luis', 'Omar', 'Abel', 'Amir', 'Eli', 'Max', 'Sam', 'Ben', 'Tom', 'Dan', 'Jay']
    name_female = ['Lily', 'Emma', 'Ella', 'Anna', 'Ava', 'Mia', 'Zoe', 'Amy', 'Ivy', 'Eva', 'Chloe', 'Kate', 'Luna', 'Ruby', 'Leah', 'Nina', 'Rosa', 'Sara', 'Lila', 'Lucy', 'Jade', 'Rose', 'Tina', 'Beth', 'Noah', 'Cleo', 'Gina', 'Tara', 'Mary', 'Joan', 'Dora', 'Erin', 'Ruth', 'Elle', 'Cara', 'Dana', 'Lana', 'Iris', 'Lacy', 'Tess', 'Joy', 'Bea', 'Meg', 'Kim', 'Liz', 'Mae', 'Faye']
    #name = name_male + name_female
    name = list(set(name_male1 + name_male2))
    
    obj = list(set(obj1 + obj2))
    
    name = [e for e in set(name) if len(tokenizer_llama.encode(e)) == 2 and len(tokenizer_qwen.encode(e)) == 1]
    career = [e for e in set(career) if len(tokenizer_llama.encode(e)) == 2 and len(tokenizer_qwen.encode(e)) == 1]
    obj = [e for e in set(obj) if len(tokenizer_llama.encode(e)) == 2 and len(tokenizer_qwen.encode(e)) == 1]
    nation = [e for e in set(nation) if len(tokenizer_llama.encode(e)) == 2 and len(tokenizer_qwen.encode(e)) == 1]
    city = [e for e in set(city) if len(tokenizer_llama.encode(e)) == 2 and len(tokenizer_qwen.encode(e)) == 1]
    
    elements = {}
    elements["name"] = name
    elements["name_female"] = name_female
    elements["career"] = career
    elements["obj"] = obj
    elements["nation"] = nation
    elements["city"] = city
    return elements


def test_story(story, ents):
    output = True
    toks = tokenizer_llama.tokenize(story)
    toks = [t.replace("Ġ", "") for t in toks]
    for ent in ents:
        try: 
            ind = toks.index(ent)
        except ValueError:
            output = False
    if """\"""" in story:
        output = False
    if "{" in story or "*" in story or "<<<" in story:
        output = False
    return output


def data_generation(elements, sdout, nb_arg=3, nb_sample=1000, datai=0):
    name = elements["name"]
    random.shuffle(name)
    name_female = elements["name_female"]
    random.shuffle(name_female)
    career = elements["career"]
    obj = elements["obj"]
    nation = elements["nation"]
    city = elements["city"]
    
    nb_name = len(name)
    nb_obj = len(obj)
    nb_city = len(city)
    nb_nation = len(nation)
    nb_career = len(career)

    print(f"nb_name: {nb_name}\n nb_obj: {nb_obj}\n nb_city: {nb_city}\n nb_nation: {nb_nation}\n nb_career: {nb_career}")
    ###Template###
    
    temp_space = " The {ENT} is manufactured in {NA1} and designed in {NA2}, and it is exported to {NA3}, but it is banned in {NA4} ."
    temp_space_p_cr = " The {ENT} is banned in {NA4} and manufactured in {NA1}, and it is designed to {NA2} and  exported in {NA3} ."
    temp_space_r_cr = " The {ENT} is introduced in {NA1} and is popular in {NA2}, and it is assembled in {NA3} and used in {NA4} ."
    
    temp_city = " {NAME} was born in {CITY1} and currently lives in {CITY2}, he loves {CITY3} and dislike {CITY4} ."
    temp_city_p_cr = " {NAME} loves {CITY3} and was born in {CITY1}, he dislikes {CITY4} and currently lives in {CITY2} ."
    temp_city_r_cr = " {NAME} works in {CITY1} and studies in {CITY2}, he travels to {CITY3} and invests in {CITY4} ."

    temp_create = " {NAME} created the {OBJ1} and also bought the {OBJ2}, he sold the {OBJ3}, and his favorite object is the {OBJ4} ."
    temp_create_p_cr = " {NAME} sold the {OBJ3} and created the {OBJ1}, his favorite object is the {OBJ4} and he bought the {OBJ2} ."
    temp_create_r_cr = " {NAME} studied the {OBJ1} and lost the {OBJ2}, he painted the {OBJ3}, and invented the {OBJ4} ."
    
    temp_relation = " {NAME} is married to {PER1} and has a child named {PER2}, he was taught by {PER3} and works under {PER4} ."
    temp_relation_p_cr = " {NAME} was taught by {PER3} and married to {PER1}, he works under {PER4} and has a child named {PER2} ."
    temp_relation_r_cr = " {NAME} lives next to {PER1} and shares room with {PER2}, he works together with {PER3} and trains {PER4} ."

    temp_job = " {NAME} currently works as a {JOB1} and dreams of becoming a {JOB2}, he previously worked as a {JOB3}, and he dislikes being a {JOB4} ."
    temp_job_p_cr = " {NAME} previously worked as a {JOB3} and currently works as a {JOB1}, he dislikes being a {JOB4} and he dreams of becoming a {JOB2} ."
    temp_job_r_cr = " {NAME} has retired as a {JOB1} and now volunteers as a {JOB2}, he is applying for a {JOB3}, and he works part time as a {JOB4} ."
    ###

    ###Table Content###
    tab_space = "| {ENT} | {NA1} | {NA2} | {NA3} | {NA4} |"
    tab_space_p_cr = "| {ENT} | {NA4} | {NA1} | {NA2} | {NA3} |"
    tab_space_r_cr = "| {ENT} | {NA1} | {NA2} | {NA3} | {NA4} |"

    tab_city = "| {NAME} | {CITY1} | {CITY2} | {CITY3} | {CITY4} |"
    tab_city_p_cr = "| {NAME} | {CITY3} | {CITY1} | {CITY4} | {CITY2} |" 
    tab_city_r_cr = "| {NAME} | {CITY1} | {CITY2} | {CITY3} | {CITY4} |"

    tab_create = "| {NAME} | {OBJ1} | {OBJ2} | {OBJ3} | {OBJ4} |"
    tab_create_p_cr = "| {NAME} | {OBJ3} | {OBJ1} | {OBJ4} | {OBJ2} |"
    tab_create_r_cr = "| {NAME} | {OBJ1} | {OBJ2} | {OBJ3} | {OBJ4} |"

    tab_relation = "| {NAME} | {PER1} | {PER2} | {PER3} | {PER4} |"
    tab_relation_p_cr = "| {NAME} | {PER3} | {PER1} | {PER4} | {PER2} |"
    tab_relation_r_cr = "| {NAME} | {PER1} | {PER2} | {PER3} | {PER4} |"

    tab_job = "| {NAME} | {JOB1} | {JOB2} | {JOB3} | {JOB4} |"
    tab_job_p_cr = "| {NAME} | {JOB3} | {JOB1} | {JOB4} | {JOB2} |"
    tab_job_r_cr = "| {NAME} | {JOB1} | {JOB2} | {JOB3} | {JOB4} |"
    ###

    ###Table Head###
    h_space = "| Product | Manufactured in | Designed in | Exported to | Banned in |"
    h_space_p_cr = "| Product | Banned in | Manufactured in | Exported to | Designed in |"
    h_space_r_cr = "| Product | Introduced in | Popular in | Assembled in | Used in |"
    
    h_city = "| Name | Birthplace | Lived City | Loved City| Disliked City |"
    h_city_p_cr = "| Name | Loved City | Birthplace | Disliked City | Lived City |"
    h_city_r_cr = "| Name | Working City | Studied City | Traveled City| Invested City |"

    h_create = "| Name | Created Object | Bought Object | Sold Object| Favorite Object |"
    h_create_p_cr = "| Name | Sold Object | Created Object | Favorite Object | Bought Object |"
    h_create_r_cr = "| Name | Studied Object | Lost Object | Painted Object| Invented Object |"

    h_relation = "| Name | Spouse | Child | Teacher | Boss |"
    h_relation_p_cr = "| Name | Teacher | Spouse | Boss | Child |"
    h_relation_r_cr = "| Name | Neighbor  | Roomate | Coworker | Apprentice |"

    h_job = "| Name | Current Job | Dream Job | Previous Job | Disliked Job |"
    h_job_p_cr = "| Name | Previous Job | Current Job | Disliked Job | Dream Job |"
    h_job_r_cr = "| Name | Retired Job  | Volunteered Job | Applied Job | Part Time Job |"
    ###
    
    rows_space = [] 
    rows_city = []
    rows_create = []
    rows_relation = []
    rows_job = []
    
    for _ in tqdm(range(nb_sample)):
        name_rand = random.sample(name, nb_arg)
        name_rand1 = random.sample(name_female, nb_arg)
        name_rand2 = random.sample(name, nb_arg)
        name_rand3 = random.sample(name, nb_arg)
        name_rand4 = random.sample(name, nb_arg)
        while nb_arg * 5 != len(set(name_rand + name_rand1 + name_rand2 + name_rand3 + name_rand4)):
            name_rand = random.sample(name, nb_arg)
            name_rand1 = random.sample(name_female, nb_arg)
            name_rand2 = random.sample(name, nb_arg)
            name_rand3 = random.sample(name, nb_arg)
            name_rand4 = random.sample(name, nb_arg)
        
        career_rand1 = random.sample(career, nb_arg)
        career_rand2 = random.sample(career, nb_arg)
        career_rand3 = random.sample(career, nb_arg)
        career_rand4 = random.sample(career, nb_arg)
        while nb_arg * 4 != len(set(career_rand1 + career_rand2 + career_rand3 + career_rand4)):
            career_rand1 = random.sample(career, nb_arg)
            career_rand2 = random.sample(career, nb_arg)
            career_rand3 = random.sample(career, nb_arg)
            career_rand4 = random.sample(career, nb_arg)

        city_rand1 = random.sample(city, nb_arg)
        city_rand2 = random.sample(city, nb_arg)
        city_rand3 = random.sample(city, nb_arg)
        city_rand4 = random.sample(city, nb_arg)
        while nb_arg * 4 != len(set(city_rand1 + city_rand2 + city_rand3 + city_rand4)):
            city_rand1 = random.sample(city, nb_arg)
            city_rand2 = random.sample(city, nb_arg)
            city_rand3 = random.sample(city, nb_arg)
            city_rand4 = random.sample(city, nb_arg)

        nation_rand1 = random.sample(nation, nb_arg)
        nation_rand2 = random.sample(nation, nb_arg)
        nation_rand3 = random.sample(nation, nb_arg)
        nation_rand4 = random.sample(nation, nb_arg)
        while nb_arg * 4 != len(set(nation_rand1 + nation_rand2 + nation_rand3 + nation_rand4)):
            nation_rand1 = random.sample(nation, nb_arg)
            nation_rand2 = random.sample(nation, nb_arg)
            nation_rand3 = random.sample(nation, nb_arg)
            nation_rand4 = random.sample(nation, nb_arg)

        obj_rand = random.sample(obj, nb_arg)
        obj_rand1 = random.sample(obj, nb_arg)
        obj_rand2 = random.sample(obj, nb_arg)
        obj_rand3 = random.sample(obj, nb_arg)
        obj_rand4 = random.sample(obj, nb_arg)
        while nb_arg * 5 != len(set(obj_rand + obj_rand1 + obj_rand2 + obj_rand3 + obj_rand4)):
            obj_rand = random.sample(obj, nb_arg)
            obj_rand1 = random.sample(obj, nb_arg)
            obj_rand2 = random.sample(obj, nb_arg)
            obj_rand3 = random.sample(obj, nb_arg)
            obj_rand4 = random.sample(obj, nb_arg)

        segs_space, segs_space_p_cr, segs_space_r_cr = [], [], []
        segs_city, segs_city_p_cr, segs_city_r_cr = [], [], []
        segs_create, segs_create_p_cr, segs_create_r_cr = [], [], []
        segs_relation, segs_relation_p_cr, segs_relation_r_cr = [], [], []
        segs_job, segs_job_p_cr, segs_job_r_cr = [], [], []

        t_segs_space, t_segs_space_p_cr, t_segs_space_r_cr = [], [], []
        t_segs_city, t_segs_city_p_cr, t_segs_city_r_cr = [], [], []
        t_segs_create, t_segs_create_p_cr, t_segs_create_r_cr = [], [], []
        t_segs_relation, t_segs_relation_p_cr, t_segs_relation_r_cr = [], [], []
        t_segs_job, t_segs_job_p_cr, t_segs_job_r_cr = [], [], []
        
        for i in range(nb_arg):
            ##space dataset##
            seg = temp_space.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            segs_space.append(seg)
            seg = temp_space_p_cr.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            segs_space_p_cr.append(seg)
            seg = temp_space_r_cr.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            segs_space_r_cr.append(seg)
            
            seg = tab_space.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            t_segs_space.append(seg)
            seg = tab_space_p_cr.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            t_segs_space_p_cr.append(seg)
            seg = tab_space_r_cr.replace("{ENT}", obj_rand[i]).replace("{NA1}", nation_rand1[i]).replace("{NA2}", nation_rand2[i]).replace("{NA3}", nation_rand3[i]).replace("{NA4}", nation_rand4[i])
            t_segs_space_r_cr.append(seg)
            ##
            
            ##city dataset##
            seg = temp_city.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            segs_city.append(seg)
            seg = temp_city_p_cr.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            segs_city_p_cr.append(seg)
            seg = temp_city_r_cr.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            segs_city_r_cr.append(seg)
            
            seg = tab_city.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            t_segs_city.append(seg)
            seg = tab_city_p_cr.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            t_segs_city_p_cr.append(seg)
            seg = tab_city_r_cr.replace("{NAME}", name_rand[i]).replace("{CITY1}", city_rand1[i]).replace("{CITY2}", city_rand2[i]).replace("{CITY3}", city_rand3[i]).replace("{CITY4}", city_rand4[i])
            t_segs_city_r_cr.append(seg)
            ##

            ##create dataset##
            seg = temp_create.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            segs_create.append(seg)
            seg = temp_create_p_cr.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            segs_create_p_cr.append(seg)
            seg = temp_create_r_cr.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            segs_create_r_cr.append(seg)

            seg = tab_create.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            t_segs_create.append(seg)
            seg = tab_create_p_cr.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            t_segs_create_p_cr.append(seg)
            seg = tab_create_r_cr.replace("{NAME}", name_rand[i]).replace("{OBJ1}", obj_rand1[i]).replace("{OBJ2}", obj_rand2[i]).replace("{OBJ3}", obj_rand3[i]).replace("{OBJ4}", obj_rand4[i])
            t_segs_create_r_cr.append(seg)
            ##

            ##relation dataset##
            seg = temp_relation.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            segs_relation.append(seg)
            seg = temp_relation_p_cr.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            segs_relation_p_cr.append(seg)
            seg = temp_relation_r_cr.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            segs_relation_r_cr.append(seg)

            seg = tab_relation.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            t_segs_relation.append(seg)
            seg = tab_relation_p_cr.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            t_segs_relation_p_cr.append(seg)
            seg = tab_relation_r_cr.replace("{NAME}", name_rand[i]).replace("{PER1}", name_rand1[i]).replace("{PER2}", name_rand2[i]).replace("{PER3}", name_rand3[i]).replace("{PER4}", name_rand4[i])
            t_segs_relation_r_cr.append(seg)
            ##

            ##job dataset##
            seg = temp_job.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            segs_job.append(seg)
            seg = temp_job_p_cr.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            segs_job_p_cr.append(seg)
            seg = temp_job_r_cr.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            segs_job_r_cr.append(seg)

            seg = tab_job.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            t_segs_job.append(seg)
            seg = tab_job_p_cr.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            t_segs_job_p_cr.append(seg)
            seg = tab_job_r_cr.replace("{NAME}", name_rand[i]).replace("{JOB1}", career_rand1[i]).replace("{JOB2}", career_rand2[i]).replace("{JOB3}", career_rand3[i]).replace("{JOB4}", career_rand4[i])
            t_segs_job_r_cr.append(seg)
            ##

        ##space dataset##
        input_space = " ".join(segs_space)
        input_space_p_cr = " ".join(segs_space_p_cr)
        input_space_r_cr = " ".join(segs_space_r_cr)
        t_input_space = "".join([h_space] + t_segs_space)
        t_input_space_p_cr = "".join([h_space_p_cr] + t_segs_space_p_cr)
        t_input_space_r_cr = "".join([h_space_r_cr] + t_segs_space_r_cr)
        ##
        ##create dataset##
        input_create = " ".join(segs_create)
        input_create_p_cr = " ".join(segs_create_p_cr)
        input_create_r_cr = " ".join(segs_create_r_cr)
        t_input_create = "".join([h_create] + t_segs_create)
        t_input_create_p_cr = "".join([h_create_p_cr] + t_segs_create_p_cr)
        t_input_create_r_cr = "".join([h_create_r_cr] + t_segs_create_r_cr)
        ##
        ##city dataset##
        input_city = " ".join(segs_city)
        input_city_p_cr = " ".join(segs_city_p_cr)
        input_city_r_cr = " ".join(segs_city_r_cr)
        t_input_city = "".join([h_city] + t_segs_city)
        t_input_city_p_cr = "".join([h_city_p_cr] + t_segs_city_p_cr)
        t_input_city_r_cr = "".join([h_city_r_cr] + t_segs_city_r_cr)
        ##
        ##relation dataset##
        input_relation = " ".join(segs_relation)
        input_relation_p_cr = " ".join(segs_relation_p_cr)
        input_relation_r_cr = " ".join(segs_relation_r_cr)
        t_input_relation = "".join([h_relation] + t_segs_relation)
        t_input_relation_p_cr = "".join([h_relation_p_cr] + t_segs_relation_p_cr)
        t_input_relation_r_cr = "".join([h_relation_r_cr] + t_segs_relation_r_cr)
        ##
        ##job dataset##
        input_job = " ".join(segs_job)
        input_job_p_cr = " ".join(segs_job_p_cr)
        input_job_r_cr = " ".join(segs_job_r_cr)
        t_input_job = "".join([h_job] + t_segs_job)
        t_input_job_p_cr = "".join([h_job_p_cr] + t_segs_job_p_cr)
        t_input_job_r_cr = "".join([h_job_r_cr] + t_segs_job_r_cr)
        ##
        
        ##space dataset##
        row = {}
        row["ents"] = obj_rand
        row["atts1"] = nation_rand1
        row["atts2"] = nation_rand2
        row["atts3"] = nation_rand3
        row["atts4"] = nation_rand4
        
        row["input"] = input_space
        row["input_p_cr"] = input_space_p_cr
        row["input_r_cr"] = input_space_r_cr
        row["t_input"] = t_input_space
        row["t_input_p_cr"] = t_input_space_p_cr
        row["t_input_r_cr"] = t_input_space_r_cr
        
        s = gen_story_via_table(row["t_input"], row["input"], [e.strip() for e in h_space.split("|") if e != ""])
        s_p_cr = gen_story_via_table(row["t_input_p_cr"], row["input_p_cr"], [e.strip() for e in h_space_p_cr.split("|") if e != ""])
        s_r_cr = gen_story_via_table(row["t_input_r_cr"], row["input_r_cr"], [e.strip() for e in h_space_r_cr.split("|") if e != ""])        
        
        row["s_input"] = s
        row["s_input_p_cr"] = s_p_cr
        row["s_input_r_cr"] = s_r_cr
        rows_space.append(row)
        ##
        
        ##create dataset##
        row = {}
        row["ents"] = name_rand
        row["atts1"] = obj_rand1
        row["atts2"] = obj_rand2
        row["atts3"] = obj_rand3
        row["atts4"] = obj_rand4
        
        row["input"] = input_create
        row["input_p_cr"] = input_create_p_cr
        row["input_r_cr"] = input_create_r_cr
        row["t_input"] = t_input_create
        row["t_input_p_cr"] = t_input_create_p_cr
        row["t_input_r_cr"] = t_input_create_r_cr
        
        s = gen_story_via_table(row["t_input"], row["input"], [e.strip() for e in h_create.split("|") if e != ""])
        s_p_cr = gen_story_via_table(row["t_input_p_cr"], row["input_p_cr"], [e.strip() for e in h_create_p_cr.split("|") if e != ""])
        s_r_cr = gen_story_via_table(row["t_input_r_cr"], row["input_r_cr"], [e.strip() for e in h_create_r_cr.split("|") if e != ""])        
        
        row["s_input"] = s
        row["s_input_p_cr"] = s_p_cr
        row["s_input_r_cr"] = s_r_cr
        rows_create.append(row)
        ##
        ##city dataset##
        row = {}
        row["ents"] = name_rand
        row["atts1"] = city_rand1
        row["atts2"] = city_rand2
        row["atts3"] = city_rand3
        row["atts4"] = city_rand4
        
        row["input"] = input_city
        row["input_p_cr"] = input_city_p_cr
        row["input_r_cr"] = input_city_r_cr
        row["t_input"] = t_input_city
        row["t_input_p_cr"] = t_input_city_p_cr
        row["t_input_r_cr"] = t_input_city_r_cr
        
        s = gen_story_via_table(row["t_input"], row["input"], [e.strip() for e in h_city.split("|") if e != ""])
        s_p_cr = gen_story_via_table(row["t_input_p_cr"], row["input_p_cr"], [e.strip() for e in h_city_p_cr.split("|") if e != ""])
        s_r_cr = gen_story_via_table(row["t_input_r_cr"], row["input_r_cr"], [e.strip() for e in h_city_r_cr.split("|") if e != ""])        
        
        row["s_input"] = s
        row["s_input_p_cr"] = s_p_cr
        row["s_input_r_cr"] = s_r_cr
        rows_city.append(row)
        ##
        
        ##relation dataset##
        row = {}
        row["ents"] = name_rand
        row["atts1"] = name_rand1
        row["atts2"] = name_rand2
        row["atts3"] = name_rand3
        row["atts4"] = name_rand4
        
        row["input"] = input_relation
        row["input_p_cr"] = input_relation_p_cr
        row["input_r_cr"] = input_relation_r_cr
        row["t_input"] = t_input_relation
        row["t_input_p_cr"] = t_input_relation_p_cr
        row["t_input_r_cr"] = t_input_relation_r_cr
        
        s = gen_story_via_table(row["t_input"], row["input"], [e.strip() for e in h_relation.split("|") if e != ""])
        s_p_cr = gen_story_via_table(row["t_input_p_cr"], row["input_p_cr"], [e.strip() for e in h_relation_p_cr.split("|") if e != ""])
        s_r_cr = gen_story_via_table(row["t_input_r_cr"], row["input_r_cr"], [e.strip() for e in h_relation_r_cr.split("|") if e != ""])        
        
        row["s_input"] = s
        row["s_input_p_cr"] = s_p_cr
        row["s_input_r_cr"] = s_r_cr
        rows_relation.append(row)
        ##
        
        ##job dataset##
        row = {}
        row["ents"] = name_rand
        row["atts1"] = career_rand1
        row["atts2"] = career_rand2
        row["atts3"] = career_rand3
        row["atts4"] = career_rand4
        
        row["input"] = input_job
        row["input_p_cr"] = input_job_p_cr
        row["input_r_cr"] = input_job_r_cr
        row["t_input"] = t_input_job
        row["t_input_p_cr"] = t_input_job_p_cr
        row["t_input_r_cr"] = t_input_job_r_cr
        
        s = gen_story_via_table(row["t_input"], row["input"], [e.strip() for e in h_job.split("|") if e != ""])
        s_p_cr = gen_story_via_table(row["t_input_p_cr"], row["input_p_cr"], [e.strip() for e in h_job_p_cr.split("|") if e != ""])
        s_r_cr = gen_story_via_table(row["t_input_r_cr"], row["input_r_cr"], [e.strip() for e in h_job_r_cr.split("|") if e != ""])        
        
        row["s_input"] = s
        row["s_input_p_cr"] = s_p_cr
        row["s_input_r_cr"] = s_r_cr
        rows_job.append(row)
        ##
        
    fout_space = open(sdout + f"space_tts_{datai}.jsonl", "w")
    fout_city = open(sdout + f"city_tts_{datai}.jsonl", "w")
    fout_create = open(sdout + f"create_tts_{datai}.jsonl", "w")
    fout_relation = open(sdout + f"relation_tts_{datai}.jsonl", "w")
    fout_job = open(sdout + f"job_tts_{datai}.jsonl", "w")

    
    for row in rows_space:
        fout_space.write(json.dumps(row))
        fout_space.write("\n")
    
    for row in rows_city:
        fout_city.write(json.dumps(row))
        fout_city.write("\n")
        
    for row in rows_create:
        fout_create.write(json.dumps(row))
        fout_create.write("\n")
    
    for row in rows_relation:
        fout_relation.write(json.dumps(row))
        fout_relation.write("\n")
    
    for row in rows_job:
        fout_job.write(json.dumps(row))
        fout_job.write("\n")
    
    fout_space.close()
    fout_city.close()
    fout_create.close()
    fout_relation.close()
    fout_job.close()
    
if __name__ == "__main__":
    nb_sample = int(sys.argv[1])
    datai = int(sys.argv[2])
    elements = load_elements()
    sdout = "/work01/daiqin/activation_patching/poe_project_copy/data/data_table/"
    data_generation(elements, sdout, nb_arg=3, nb_sample=nb_sample, datai=datai)
    
    
