# LLM Text Forensics

Outil de recherche pour la détection de texte généré par IA, combinant deux
familles de méthodes complémentaires :

- **Détection de watermark** — recherche la signature statistique laissée
  volontairement dans le texte par un modèle qui applique un watermark au
  moment de la génération (ex. SynthID). Fiable quand elle détecte un
  watermark, mais ne fonctionne que si le modèle générateur en a posé un.
- **Détection statistique (zero-shot)** — ne suppose aucun watermark ;
  compare la perplexité du texte entre deux modèles de langage (un
  "observateur" et un "performeur") pour estimer s'il a été produit par un
  LLM (méthode Binoculars).

Les résultats par méthode sont combinés par `aggregator/` en un verdict
unique avec un niveau de confiance, exposé via une interface Gradio
(`demo/app.py`).

## Architecture

```
common/         types partagés (DetectionResult, ...)
watermark/      détecteurs de watermark (SynthID, extensible)
statistical/    détecteur statistique Binoculars
aggregator/     combine les résultats des détecteurs en un verdict unique
demo/           interface Gradio (coller/importer un texte, jauge de risque)
```

## Installation

```bash
pip install -e .[demo]
```

## Lancer la démo

```bash
python -m demo.app
```

Ouvrez ensuite http://127.0.0.1:7860.

## Limites connues

La détection statistique (Binoculars) cible la signature d'un texte
échantillonné brut par un LLM. Un texte généré avec beaucoup de RAG
(passages issus de documents réels, cités ou reformulés) ou fortement
retravaillé après génération se rapproche statistiquement d'un texte humain
et peut échapper à la détection. Les seuils de décision fournis sont en
outre calibrés pour une paire de modèles précise (falcon-7b /
falcon-7b-instruct, cf. `statistical/binoculars/config.py`) et ne se
transposent pas automatiquement à d'autres modèles ou langues.

## Crédits

Ce projet s'appuie sur les implémentations et travaux de recherche suivants :

- **[Binoculars](https://github.com/ahans30/Binoculars)** — Hans, Abhimanyu,
  et al. *"Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-
  Generated Text."* ([arXiv:2401.12070](https://arxiv.org/abs/2401.12070)).
  Le détecteur statistique de ce dépôt (`statistical/binoculars/`) est porté
  depuis leur implémentation de référence.
- **[SynthID Text](https://github.com/google-deepmind/synthid-text)** —
  Google DeepMind. Le détecteur de watermark (`watermark/synthid/`)
  s'appuie sur leur implémentation de référence du schéma de watermarking
  SynthID.

## Licence

Apache-2.0, voir [LICENSE](LICENSE).
