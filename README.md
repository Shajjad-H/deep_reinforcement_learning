### mise en route
`conda env create -f environment.yml` 

#### ou installation manuelle

`conda install matplotlib scikit-image pytorch torchvision -c pytorch`

`conda install -c conda-forge gym`


et installer vizdoomgym en suivant le lien `https://github.com/shakenes/vizdoomgym`

#### entrainement des models

Pour entrainner les modeles il suffit de lancer le fichier `cart_pole_train.py` pour l'agent Cart Pole.
Pour enregistrer le modele avec la meilleur performance il faut mettre `save_model` à True dans la config. Et pour lancer l'evaluation de modele il suffit de lancer le fichier `cart_pole_eval.py`.