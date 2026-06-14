# Projet Deep Learning EMSI — 2025/2026

**Réalisé par : Hibatallah Baetouty**  
**Établissement : EMSI | Module : Deep Learning | Année : 2025/2026**

---

## Idée générale du projet

Ce projet explore trois grandes familles d'architectures de deep learning appliquées au domaine de la **santé et médecine**, à travers PyTorch :

- **Partie I — MLP** : classification de tumeurs mammaires (données tabulaires) avec le dataset *Breast Cancer Wisconsin*. L'objectif est de maîtriser les bases de PyTorch (`nn.Module`, initialisations, sauvegarde/chargement de modèles) et d'évaluer l'impact des stratégies d'initialisation sur la convergence.

- **Partie II — CNN** : détection de pneumonie sur radiographies thoraciques (*PneumoniaMNIST*). On implémente une architecture LeNet adaptée, on étudie les hyperparamètres CNN (padding, stride, pooling, filtres), on visualise les feature maps, et on compare les performances MLP vs CNN sur données images.

- **Partie III — RNN / LSTM / GRU / Seq2Seq** : analyse de reviews médicales (*UCI Drug Reviews*). On explore les modèles récurrents (BPTT, gradient clipping), l'architecture encodeur-décodeur Seq2Seq avec teacher forcing, et les stratégies de décodage (glouton, Beam Search) avec évaluation BLEU.

Une **question transversale finale** relie les trois parties en analysant pourquoi le deep learning décline ses architectures différemment selon la géométrie des données (tabulaire, image, séquentielle).

---

## Structure du projet

```
Projet/
├── partie1_MLP/
│   └── mlp_breast_cancer.ipynb       # MLP sur données tabulaires
├── partie2_CNN/
│   └── cnn_pneumonia.ipynb           # CNN sur radiographies médicales
├── partie3_RNN_Seq2Seq/
│   └── rnn_seq2seq_drug_reviews.ipynb # RNN/LSTM/GRU/Seq2Seq sur texte médical
└── requirements.txt
```

## Datasets

| Partie | Dataset | Accès |
|--------|---------|-------|
| I — MLP | Breast Cancer Wisconsin | `sklearn.datasets.load_breast_cancer()` |
| II — CNN | PneumoniaMNIST | `pip install medmnist` |
| III — RNN/Seq2Seq | UCI Drug Reviews | HuggingFace `lewtun/drug-reviews` |

## Environnement recommandé : Google Colab (GPU T4)

### Exécution locale
```bash
pip install -r requirements.txt
jupyter notebook
```

## Contenu par partie

### Partie I — MLP & PyTorch
- Concepts : `nn.Module`, `state_dict`, `named_parameters`, device
- 2 implémentations : `nn.Sequential` + classe personnalisée
- 3 stratégies d'initialisation : Gaussienne, Constante, Xavier
- Entraînement + sauvegarde du meilleur modèle
- Métriques : Accuracy, Precision, Recall, F1, Matrice de confusion

### Partie II — CNN & Vision Médicale
- Corrélation croisée 2D et pooling manuels
- Architecture LeNet adaptée à la détection de pneumonie
- Étude hyperparamètres : padding, stride, pooling, filtres, conv 1×1
- Visualisation des feature maps
- Comparaison MLP vs CNN

### Partie III — RNN / LSTM / GRU / Seq2Seq
- Modèles de langage, perplexité
- RNN, LSTM, GRU avec BPTT et gradient clipping
- Comparaison des 3 architectures récurrentes
- Seq2Seq encodeur-décodeur avec teacher forcing
- Décodage glouton et Beam Search
- Évaluation BLEU

## Question transversale finale

> Comment le deep learning adapte-t-il ses architectures à la structure des données
> — tabulaire, image, séquentielle — et pourquoi un même paradigme d'apprentissage
> supervisé doit-il être décliné différemment selon la géométrie, la dépendance
> locale, la temporalité et la représentation des données ?

### Réponse

Le deep learning repose sur un paradigme unifié : apprendre une fonction paramétrique
`f_θ : X → Y` par minimisation d'une perte via rétropropagation. Pourtant, ce cadre
commun se décline en architectures radicalement différentes selon la **géométrie
intrinsèque** des données. Ce projet illustre trois cas distincts.

---

#### 1. Données tabulaires → MLP (Partie I)

Le dataset Breast Cancer Wisconsin est constitué de **30 features numériques continues
et indépendantes**, sans structure spatiale ni temporelle. Chaque feature (rayon, texture,
périmètre…) est globale et échangeable. Le **MLP** est naturellement adapté : chaque
neurone de la première couche traite l'intégralité du vecteur d'entrée et apprend des
combinaisons linéaires arbitraires entre features.

- **Géométrie** : espace vectoriel R^30, sans voisinage défini
- **Inductive bias** : aucun — le MLP n'impose aucune hypothèse de structure
- **Résultat** : accuracy ~97% avec une architecture 30→64→32→16→2 et initialisation Xavier

L'absence d'inductive bias est un avantage ici : les corrélations entre features
(ex. rayon et périmètre) sont apprises librement. En revanche, le MLP ne généralise
pas aux données où la position relative des éléments a un sens.

---

#### 2. Données images → CNN (Partie II)

Une radiographie thoracique (PneumoniaMNIST 28×28) n'est pas un vecteur de 784 valeurs
indépendantes : les pixels **voisins sont corrélés**, et un pattern diagnostique
(infiltrat, opacité) peut apparaître **n'importe où** dans l'image. Le **CNN** encode
ces deux propriétés :

- **Localité** : chaque filtre ne perçoit qu'une région (champ récepteur 5×5), ce qui
  correspond à la structure physique des pathologies
- **Invariance par translation** : le même filtre est convoluté sur toute l'image —
  une opacité en haut à gauche ou en bas à droite active le même détecteur
- **Hiérarchie** : Conv1 détecte bords/contrastes, Conv2 détecte textures/formes

- **Géométrie** : grille 2D avec structure locale (H×W×C)
- **Inductive bias** : localité + partage des poids + invariance par translation
- **Résultat** : accuracy ~88% sur test, avec **10× moins de paramètres** que le MLP
  équivalent (61k vs 214k)

Un MLP appliqué aux mêmes images (pixels aplatis) ignore le voisinage spatial et
nécessite 3.5× plus de paramètres pour des performances inférieures.

---

#### 3. Données séquentielles → RNN / LSTM / GRU / Seq2Seq (Partie III)

Une review médicale ("*This drug did not help my pain at all*") est une séquence de
tokens où **l'ordre et le contexte temporel sont essentiels** : "not help" n'est pas
équivalent à "help not". De plus, la dépendance entre tokens peut s'étendre sur
plusieurs phrases. Le MLP et le CNN ignorent cette temporalité.

- **RNN** : maintient un état caché `h_t = φ(W_xh·x_t + W_hh·h_{t-1})` qui résume
  le contexte passé. Mais le gradient évanescent empêche de capturer les dépendances
  lointaines (> 10-15 tokens)
- **LSTM** : introduit un état de cellule `c_t` avec portes d'oubli/entrée/sortie,
  créant un chemin de gradient stable sur de longues séquences (+4-6% accuracy vs RNN)
- **GRU** : simplifie le LSTM (2 portes vs 3) avec des performances équivalentes et
  30% de paramètres en moins — meilleur compromis coût/performance
- **Seq2Seq** : quand la sortie est elle-même une séquence, l'encodeur condense la
  source en vecteur de contexte, le décodeur le déplie token par token avec
  teacher forcing à l'entraînement et beam search à l'inférence

- **Géométrie** : séquence 1D ordonnée avec dépendances temporelles
- **Inductive bias** : récurrence (chaque état dépend du précédent) + mémoire sélective
- **Résultat** : LSTM/GRU ~70% accuracy sur sentiment médical, perplexité Seq2Seq
  descendant de 514 → 48 en 8 époques

---

#### Synthèse : pourquoi un même paradigme, trois architectures ?

| Dimension | Données tabulaires | Images | Séquences |
|-----------|-------------------|--------|-----------|
| **Structure** | Vecteur plat, features indépendantes | Grille 2D, pixels corrélés localement | Suite ordonnée, dépendances temporelles |
| **Inductive bias clé** | Aucun (MLP universel) | Localité + translation-invariance | Récurrence + mémoire sélective |
| **Architecture** | MLP | CNN | RNN / LSTM / GRU / Seq2Seq |
| **Paramètre critique** | Initialisation (Xavier) | Padding, stride, nb filtres | Gradient clipping, nb couches |
| **Risque principal** | Surapprentissage sur petits datasets | Perte d'info spatiale (stride agressif) | Gradient évanescent (RNN simple) |

Le **paradigme supervisé reste identique** (minimiser L(f_θ(x), y) par SGD), mais
l'architecture encode des **a priori structurels** différents sur les données. Imposer
le bon inductive bias n'est pas une contrainte — c'est ce qui rend l'apprentissage
possible avec un nombre fini d'exemples : le CNN n'a pas besoin de voir une opacité
à chaque position pour la détecter partout ; le LSTM n'a pas besoin de mémoriser
chaque token pour comprendre le sens global d'une phrase.

En médecine, ce choix architectural est critique : utiliser un MLP sur des
radiographies revient à ignorer la localisation anatomique des lésions ; utiliser un
RNN simple sur de longues anamnèses revient à oublier le début du dossier patient.
L'architecture doit refléter la physique des données.
