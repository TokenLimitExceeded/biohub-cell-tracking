# biohub-cell-tracking

Final submission for the Kaggle competition [Biohub](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

Public 0.968 (29th), private 0.924 (169th).

Built on the public 0.947 notebook. Detection and linking use pilkwang's public TemporalUNet3D and SimpleNodeTransformer weights. 

Added:

- LightGBM division classifier on image features, adds divisions for orphan daughters and relinks non-orphan daughters when the score is high
- drop the 30% of geometric forks with the lowest classifier score
- add and replace edges using Trackastra scores
- relinking compensated by local tissue flow, initial matching seeded by a global displacement vote
- coordinate refinement head trained on frozen UNet features, z target halved
- when the Trackastra ILP keeps fewer than 0.95x the nodes of the original ILP, merge in the segments only the original ILP has
- line-fit smoothing (window ±4) on positions with whole-frame jumps removed
- after smoothing, push short fragments slightly away from nearby long tracks

## Files

- `notebook/` final notebook 
- `assets/prepare.py` Kaggle script that packages Trackastra
- `models/` division classifier and coordinate head weights
- `cellmot/` local experiment code
- `train/division/` division classifier training
- `train/coord_head/` coordinate head training
