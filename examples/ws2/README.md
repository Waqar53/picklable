# Real conveyor examples (WS2)

The before frames are real images from a plastic-sorting plant, from the WS2 dataset:
Marelli et al., "WS2: Weakly Supervised Segmentation using Before-After Supervision in Waste
Sorting", ICCV 2025 Workshops, https://zenodo.org/records/14793518, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Changes made here: one annotated item was removed from each before frame by inpainting (some
neighbours nudged a few pixels) and the result re-captured with camera shake, lighting drift,
noise and JPEG to make the after frame (`scripts/ws2_removal_benchmark.py`). `gt_mask.png` is
the dataset's hand-made mask of that item. The last example is a missed grasp: nothing removed.
