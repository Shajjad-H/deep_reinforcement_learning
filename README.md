# Guide d'installation et d'utilisation

Ce projet permet d’entraîner et d’évaluer des agents sur différents environnements, notamment **CartPole** et **VizDoom**.

---

## 1. Installation

### Option 1 : Création de l’environnement via `environment.yml`

```bash
conda env create -f environment.yml
```

### Option 2 : Installation manuelle

Installez les dépendances principales :

```bash
conda install matplotlib scikit-image pytorch torchvision -c pytorch
conda install -c conda-forge gym
```

Ensuite, installez **vizdoomgym** en suivant les instructions du dépôt officiel :
[https://github.com/shakenes/vizdoomgym](https://github.com/shakenes/vizdoomgym)

---

## 2. Entraînement des modèles

### CartPole

Pour entraîner l’agent CartPole :

```bash
python cart_pole_train.py
```

* Pour sauvegarder automatiquement le modèle ayant la meilleure performance, réglez la variable `save_model` sur `True` dans le fichier de configuration.

### VizDoom

Pour entraîner un agent VizDoom, utilisez les scripts correspondants à l’environnement VizDoom, selon la même logique que pour CartPole.

---

## 3. Évaluation des modèles

### CartPole

Pour évaluer un modèle entraîné :

```bash
python cart_pole_eval.py
```

### VizDoom

De même, utilisez les scripts d’évaluation spécifiques à l’environnement VizDoom.

---

## 4. Notes

* Assurez-vous que l’environnement Conda est activé avant de lancer les scripts :

```bash
conda activate <nom_de_votre_env>
```

* Les configurations et hyperparamètres sont modifiables directement dans les fichiers de configuration pour chaque environnement.


