# Projet Deep Learning EMSI — 2025/2026
**Thème transversal : Santé & Médecine**

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
