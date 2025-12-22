# Spec2vecAnalyse

## Apply spectractor

Pour appliqué spectractor à un dossier de simu, on peut faire :

```python
python Spec2vecAnalyse/apply_spectractor.py <nom_dossier>
```

Le <nom_dossier> doit être dans le dossier `results/output_simu`.

Cela va créer:
* *pred_Spectractor_x_x_0e+00*, un dossier qui contient les extractions en npy (les intensité, pour les longueurs d'ondes entre 300 et 1100 par pas de 1)
* *spectractor_exceptions.json*, un json avec comme clefs les noms des images qui ont plantés, avec les traceback associé.
* *spectrum_fits*, le résultat en fits de l'extraction donné par spectractor
