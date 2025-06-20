
# 📰 TruthScope 𖣨🔎

Ce projet vise à détecter automatiquement les **fake news** à partir de textes en utilisant un modèle BERT optimisé.  
Il s'agit d'une solution complète comprenant un backend en Python (FastAPI) et une interface utilisateur simple (HTML/CSS/JS).

---

## 📊 Dataset utilisé

Le modèle a été entraîné sur un dataset construit à partir de la **fusion de trois sources de données fiables** largement utilisées dans la recherche sur la détection de fausses informations :

1. **FakeNewsNet** ([source](https://github.com/KaiDMML/FakeNewsNet))  
   - Contient deux sous-ensembles : *PolitiFact* et *GossipCop*  
   - Chaque instance est étiquetée comme `Fake` ou `True`

2. **TruthSeeker 2023**  
   - Dataset récent dédié à la détection des fake news sur les réseaux sociaux  
   - Données vérifiées manuellement, avec étiquetage binaire (`0`: fake, `1`: real)

3. **Fake vs Real News Dataset** ([source GitHub](https://github.com/HNDeshanSamarathunga/FakeNewsDetection))  
   - Contient des textes d’actualités simples classés en `fake` et `true`

👉 Toutes les données ont été **nettoyées, uniformisées et concaténées** dans un seul dataset final utilisé pour l'entraînement.  
Ce dataset fusionné contient deux colonnes principales :

- `text` : contenu textuel de la news  
- `label` : `0` pour les fake news, `1` pour les vraies

---

## 🚀 Démarrer le projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/AnisBenini/Detection_of_Fake_News.git 
cd Detection_of_Fake_News
```

### 2. Activer le backend (FastAPI)

Depuis le dossier `main`, lancez le serveur backend :

```bash
cd main
uvicorn app:app --reload
```

> ⚠️ Assurez-vous que tous les packages nécessaires sont installés (`transformers`, `fastapi`, `uvicorn`, etc.)

### 3. Lancer l’interface utilisateur

Depuis le dossier `UI`, ouvrez le fichier `index.html` avec **Live Server** (via Visual Studio Code ou tout outil similaire) :

```bash
cd ../UI
```

> Clic droit sur `index.html` → **"Open with Live Server"**

✅ L’interface est déjà connectée au backend.  
Vous pouvez saisir un texte dans l’interface : il sera envoyé automatiquement au modèle, qui renverra une prédiction (`Fake` ou `Real`) affichée instantanément à l’utilisateur.

---

## 🧠 Modèle utilisé

- **Architecture** : BERT (base uncased), modèle de référence pour le NLP

### 🛠️ Techniques d'entraînement

- **Fine-tuning** : adaptation spécifique du modèle aux données de fake news  
- **QLoRA (4-bit)** : optimisation mémoire pour l’entraînement  
- **Quantization** : réduction de la taille du modèle sans perte significative de performance

### 🧰 Outils et bibliothèques

- [Transformers](https://huggingface.co/docs/transformers/index) – par HuggingFace  
- [Datasets](https://huggingface.co/docs/datasets/)  
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – pour la quantization 4-bit  
- [PEFT](https://github.com/huggingface/peft) – pour l'entraînement efficace avec LoRA  
- [PyTorch](https://pytorch.org/) – framework de deep learning

---

## 👨‍🏫 Note à l’enseignant

**Remarque importante :**  
Les métriques globales du modèle (accuracy, f1-score,precision..) sont très élevées et démontrent une bonne performance sur l’ensemble des données.  
Cependant, lors de tests manuels via l’UI, j’ai constaté que dans certains cas particuliers j'obtiens de faux resultat 

Cela peut être dû à des limites du modèle BERT ou à la complexité sémantique de certains exemples. Pour résoudre cela, je prévois d’explorer :

- Une meilleure gestion du déséquilibre de classes
- Des techniques d’augmentation de données ou d’ensemblage
- Une piste d’amélioration envisagée est l’intégration d’un module de fact-checking, permettant de vérifier la véracité des faits évoqués dans les textes à l’aide de bases de connaissances externes ou d’API spécialisées. Je me demande s’il serait pertinent de l’ajouter à ce projet pour renforcer la robustesse du système.

Cette observation montre que, bien qu’efficace, le modèle nécessite encore des améliorations pour atteindre une robustesse optimale dans des cas réels.

Le backend et le frontend sont **complètement intégrés**.  
L’application fonctionne localement dès l’exécution de `uvicorn` et l’ouverture de l’interface.  
Toutes les étapes de traitement des données et d’entraînement du modèle sont détaillées dans le notebook `main/Main.ipynb`.
