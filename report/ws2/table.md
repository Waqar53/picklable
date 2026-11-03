| Mask source | mean IoU | IoU >= 0.75 | IoU >= 0.5 | missed grasps caught | auto-accepted | sent to review | score vs IoU (Spearman) | time / pick |
|---|---|---|---|---|---|---|---|---|
| change map + GrabCut | 0.889 | 91% | 98% | 21/21 | 88/99 at IoU 0.915 | 11 at IoU 0.67 | 0.74 | 0.8 s |
| SAM 2.1-tiny at the grasp | 0.917 | 91% | 98% | 21/21 | 90/99 at IoU 0.944 | 9 at IoU 0.65 | 0.70 | 3.8 s |
