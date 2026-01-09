import gzip
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import BPR, VBPR
from collections import Counter

DATA_FILE = "Toys_and_Games.jsonl.gz"

# seuil de filtrage pour les utilisateurs

MIN_USER_FREQ = 5

# seuil de filtrage pour les articles

MIN_ITEM_FREQ = 5 

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# taille des morceaux pour le traitement des données en mémoire

CHUNK_SIZE = 500000 





print("Comptage des lignes...")

with gzip.open(DATA_FILE, 'rt') as f:

    num_lines = sum(1 for _ in f)

print(f"Nombre de lignes à charger: {num_lines}")



print("Lecture initiale pour calculer K-Core (IDs uniquement)...")



# on ne garde que les colonnes utiles pour le K-Core

use_cols = ['user_id', 'asin']

chunks = []



with gzip.open(DATA_FILE, 'rt') as f, tqdm(total=num_lines, desc="Lecture K-Core") as pbar:

    for chunk in pd.read_json(f, lines=True, chunksize=CHUNK_SIZE):

        chunks.append(chunk[use_cols])

        pbar.update(len(chunk))



df_ids = pd.concat(chunks, ignore_index=True)

del chunks  # libérer la RAM



# filtrage K-core

def k_core_filter_ids(df, k_user, k_item, max_iter=20):

    df_core = df.copy()

    for i in range(max_iter):

        n_users = df_core['user_id'].nunique()

        n_items = df_core['asin'].nunique()

        n_rows  = len(df_core)



       #  comptage du nombre d'apparitions de chaque utilisateur et de chaque produit

        user_cnt = df_core.groupby('user_id').size()

        item_cnt = df_core.groupby('asin').size()



       # filtrage

        df_core = df_core[

            df_core['user_id'].isin(user_cnt[user_cnt >= k_user].index) &

            df_core['asin'].isin(item_cnt[item_cnt >= k_item].index)

        ]



        print(

            f"Iter {i+1}: "

            f"{n_rows} → {len(df_core)} interactions | "

            f"{n_users} → {df_core['user_id'].nunique()} users | "

            f"{n_items} → {df_core['asin'].nunique()} items"

        )



        # condition de l’arrêt 

        if df_core['user_id'].nunique() == n_users and df_core['asin'].nunique() == n_items:

            break



    return set(df_core['user_id']), set(df_core['asin'])



valid_users, valid_items = k_core_filter_ids(df_ids, MIN_USER_FREQ, MIN_ITEM_FREQ)

del df_ids  # libérer RAM



print("Lecture finale des interactions filtrées (ratings)...")



chunks = []



with gzip.open(DATA_FILE, 'rt') as f, tqdm(total=num_lines, desc="Lecture interactions finales") as pbar:

    for chunk in pd.read_json(f, lines=True, chunksize=CHUNK_SIZE):

        chunk = chunk[

            chunk['user_id'].isin(valid_users) &

            chunk['asin'].isin(valid_items)

        ]

        if len(chunk) == 0:

            pbar.update(len(chunk))

            continue



        chunks.append(chunk[['user_id', 'asin', 'rating']])

        pbar.update(len(chunk))



df_interactions_clean = pd.concat(chunks, ignore_index=True)



print(

    f"Interactions finales : {len(df_interactions_clean)} | "

    f"Users : {df_interactions_clean['user_id'].nunique()} | "

    f"Items : {df_interactions_clean['asin'].nunique()}"

)





METADATA_FILE = "meta_Toys_and_Games.jsonl.gz"

# on ne garde que les colonnes utiles pour les embeddings textuels

use_meta_cols = ['parent_asin', 'title', 'description', 'features', 'categories']



print("Lecture finale des métadonnées et filtrage K-Core...")

chunks = []

num_meta_lines = sum(1 for _ in gzip.open(METADATA_FILE, 'rt'))



with gzip.open(METADATA_FILE, 'rt') as f, tqdm(total=num_meta_lines, desc="Lecture metadata") as pbar:

    for chunk in pd.read_json(f, lines=True, chunksize=CHUNK_SIZE):

        # Filtrer seulement les items présents dans le K-Core

        chunk = chunk.rename(columns={'parent_asin': 'asin'})

        chunk = chunk[chunk['asin'].isin(valid_items)]

        if len(chunk) == 0:

            pbar.update(len(chunk))

            continue



        # remplacer les NaN par des chaînes vides

        for col in use_meta_cols[1:]:

            chunk[col] = chunk[col].fillna('').astype(str)



        # fusionne les colonnes importantes en une seule chaîne de texte

        chunk['text'] = (

            chunk['title'] + '. ' +

            chunk['description'] + '. ' +

            chunk['features'] + '. ' +

            chunk['categories'] + '. '

        ).str.strip()



        # on ne garde que l'identifiant et le texte

        chunks.append(chunk[['asin', 'text']])

        pbar.update(len(chunk))



# concaténation finale

df_meta = pd.concat(chunks, ignore_index=True)

print(f"Total items après filtrage : {len(df_meta)}")



def clean_metadata_text(text):

    # Supprime espaces multiples et trim

    text = re.sub(r'\s+', ' ', text)

    return text.strip()



df_meta['text'] = df_meta['text'].apply(clean_metadata_text)



# préparation des embeddings

ids_list = df_meta['asin'].tolist()

corpus_list = df_meta['text'].tolist()



print(f"Génération des embeddings pour {len(ids_list)} items avec {EMBEDDING_MODEL}...")

model_nlp = SentenceTransformer(EMBEDDING_MODEL)

item_embeddings = model_nlp.encode(corpus_list, show_progress_bar=True, convert_to_numpy=True)



# normalisation des vecteurs (BPR / VBPR)

item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)



print(f"Dimension des embeddings : {item_embeddings.shape}")



# tuples crée pour l’interaction utilisateur-item

feedback_tuples = list(df_interactions_clean[['user_id', 'asin', 'rating']].itertuples(index=False, name=None))

print(f"Exemple de tuple : {feedback_tuples[:5]}")

import matplotlib.pyplot as plt





num_users = df_interactions_clean['user_id'].nunique()

num_items = df_interactions_clean['asin'].nunique()

num_interactions = len(df_interactions_clean)



density = num_interactions / (num_users * num_items)

print(f"Nombre d'utilisateurs : {num_users}")

print(f"Nombre d'items : {num_items}")

print(f"Nombre d'interactions : {num_interactions}")

print(f"Densité de la matrice user-item : {density:.6f} ({density*100:.4f}%)\n")





items_with_text = set(ids_list)

items_in_ratings = set(i for _, i, _ in feedback_tuples)



print("Items avec texte :", len(items_with_text))

print("Items dans ratings :", len(items_in_ratings))

print("Items sans texte :", len(items_in_ratings - items_with_text))



print("Filtrage des ratings...")



# filtrage : garde que les interactions si l’identifiant du produit  existe dans notre liste de textes

feedback_tuples_filtered = [

    (u, i, r)

    for (u, i, r) in feedback_tuples

    if i in items_with_text

]



print(f"Ratings avant : {len(feedback_tuples)}")

print(f"Ratings après  : {len(feedback_tuples_filtered)}")



from collections import Counter



user_counts = Counter(u for u, _, _ in feedback_tuples_filtered)



#  reconstruit la liste en vérifiant que l'utilisateur a au moins 1 interaction

feedback_tuples_filtered = [

    (u, i, r)

    for (u, i, r) in feedback_tuples_filtered

    if user_counts[u] > 0

]



# on lie les embeddings aux IDs des produits

item_modality = cornac.data.ImageModality(

    features=item_embeddings,

    ids=ids_list

)



# sert à entraîner des modèles qui utilisent les descriptions

rs_embeddings = RatioSplit(

    data=feedback_tuples_filtered,

    test_size=0.2,

    rating_threshold=4.0,

    item_image=item_modality,

    seed=123,

    verbose=True,

    exclude_unknowns=True

)



# sert à entraîner les modèles de recommandation classiques

rs_base = RatioSplit(

    data=feedback_tuples_filtered,

    test_size=0.2,

    rating_threshold=4.0,

    seed=123,

    verbose=True,

    exclude_unknowns=True

)



# Métriques

metrics = [cornac.metrics.Recall(k=10), cornac.metrics.NDCG(k=10)]



print(">>> Définition des modèles...")





from cornac.models import PMF, BPR, VBPR

from cornac.eval_methods import RatioSplit

from cornac.hyperopt import GridSearch, Discrete

import cornac



# 1. Baseline Standard : PMF

pmf = PMF(k=20, max_iter=50, learning_rate=0.01, lambda_reg=0.01, verbose=True)



# 2. Baseline Ranking : BPR

bpr = BPR(k=50, max_iter=50, learning_rate=0.01, verbose=True)



# 3. VBPR

vbpr = VBPR(

    k=40,

    k2=50,

    n_epochs=6,

    learning_rate=0.005,

    lambda_w=0.05,

    lambda_b=0.05,

    verbose=True

)

print(">>> Exécution finale...")



cornac.Experiment(

    eval_method=rs_base,

    models=[pmf],

    metrics=metrics,

    user_based=True

).run()



cornac.Experiment(

    eval_method=rs_base,

    models=[bpr],

    metrics=metrics,

    user_based=True

).run()



cornac.Experiment(

    eval_method=rs_embeddings,

    models=[vbpr],

    metrics=metrics,

    user_based=True

).run()
