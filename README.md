# tp_deep_rl

https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf


### mise en route
`conda env create -f environment.yml` 

#### ou installation manuelle

`conda install matplotlib scikit-image pytorch torchvision -c pytorch`

`conda install -c conda-forge gym`

#### entrainement des models

Pour entrainner les modeles il suffit de lancer le fichier `cart_pole_train.py` pour l'agent Cart Pole.
Pour enregistrer le modele avec la meilleur performance il faut mettre `save_model` à True dans la config.