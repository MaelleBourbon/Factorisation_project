# Hybrid Recommendation System: Amazon Toys & Games

Ce projet implémente une solution de recommandation hybride combinant le **filtrage collaboratif** et le **traitement du langage naturel (NLP)**.  
L'objectif est de prédire les préférences des utilisateurs en exploitant à la fois l'historique des interactions et les métadonnées textuelles des produits.

---

## Description

Le projet utilise le jeu de données **Toys and Games** d'Amazon. 
- [Toys_and_Games.jsonl.gz]( )  
- [meta_Toys_and_Games.jsonl.gz]( ) (métadonnées des items : title, description, features, categories)

> Note : Ces fichiers sont compressés au format `.jsonl.gz`. Les colonnes importantes pour le projet sont précisées dans le script `main.py`.
 
Il compare des approches classiques de factorisation de matrice avec une approche à l'état de l'art (**VBPR**) détournée pour intégrer des **embeddings sémantiques** (extraits des descriptions de produits) au lieu des caractéristiques visuelles habituelles.

---

## Pipeline de données

1. **Chargement chunké & compressé**  
   - Lecture des fichiers `.jsonl.gz` par morceaux de 500 000 lignes pour éviter la saturation de la RAM.

2. **Filtrage K-Core (Itératif)**  
   - Application d'un filtre `K=5` (ne garder que les utilisateurs et items ayant au moins 5 interactions) pour réduire la sparsité et améliorer la fiabilité du modèle.

3. **NLP Embedding**  
   - Fusion des colonnes `title`, `description`, `features` et `categories`.  
   - Vectorisation via le modèle pré-entraîné `all-MiniLM-L6-v2` (Sentence-Transformers).  
   - Normalisation L2 des vecteurs pour une meilleure stabilité dans l'espace latent.  
   - **Alignement** : filtrage final pour s'assurer que chaque interaction possède bien un vecteur de caractéristiques textuelles correspondant.

---

## Modèles évalués

Nous utilisons la bibliothèque **Cornac** pour comparer trois types de modèles :

| Modèle | Type        | Description                                                                 |
|--------|------------|-----------------------------------------------------------------------------|
| PMF    | Collaboratif | Probabilistic Matrix Factorization : approche de base par factorisation.   |
| BPR    | Collaboratif | Bayesian Personalized Ranking : optimisation basée sur le classement relatif (ranking). |
| VBPR   | Hybride     | Visual BPR : utilise les embeddings textuels comme des "features" pour guider la factorisation. |

---

## Installation

Pour exécuter ce notebook, les bibliothèques suivantes sont nécessaires :

```bash
pip install pandas numpy sentence-transformers cornac tqdm matplotlib


---

## Résultats

L'évaluation est réalisée sur un découpage 80/20 (RatioSplit) avec un seuil de pertinence fixé à 4.0/5.

Scores après l'exécution finale du code :

| Modèle | Recall@10 | NDCG@10 |
|--------|-----------|---------|
| PMF    | 0.XXXX    | 0.XXXX  |
| BPR    | 0.XXXX    | 0.XXXX  |
| VBPR   | 0.XXXX    | 0.XXXX  |
