Auto-labeler ablation on 10 synthetic sessions (`picklabel synth --sessions 10 --seed 1`), 37 picks with gold masks; propagation off.

| Variant | mean IoU | IoU >= 0.75 | false-change px (median, % of frame) |
|---|---|---|---|
| full pipeline | 0.925 | 89% | 2.6% |
| no exposure / white-balance gains | 0.825 | 70% | 3.9% |
| no shift tolerance | 0.793 | 68% | 3.1% |
| neither | 0.607 | 39% | 13.8% |
| no GrabCut edge snapping | 0.780 | 76% | 2.6% |
