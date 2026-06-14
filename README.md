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
