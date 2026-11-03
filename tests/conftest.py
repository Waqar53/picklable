import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "picklabel_ros"))


@pytest.fixture(scope="session")
def small_cfg():
    from picklabel.data.synth import SynthConfig

    return SynthConfig(width=480, height=360, n_organic=(18, 28), n_plastic=(3, 4))


@pytest.fixture(scope="session")
def synth_session(small_cfg):
    from picklabel.data.synth import make_session

    pairs, final_after, cleared = make_session("t000", np.random.default_rng(5), small_cfg)
    return pairs, final_after, cleared


def textured(h=240, w=320, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 4)
    base = (base - base.min()) / (base.max() - base.min())
    return (40 + 180 * base).astype(np.uint8)
