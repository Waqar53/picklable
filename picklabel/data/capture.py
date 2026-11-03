"""Capture rig: record before/after pairs with the grasp point, in PickLabel's folder layout.

Set-up: phone face-down 50-60 cm above a dark tray, focus and exposure locked in the camera
app (white balance too, if it allows), streamed to the laptop (an IP-camera app URL) or
used as a camera index (Continuity Camera, USB webcam).

Keys in the window:
  b       capture BEFORE (averages --avg frames to cut sensor noise)
  click   set the grasp point on the frozen before frame
  a       capture AFTER, once the item is removed with tongs -> pair saved
  m       AFTER for a missed grasp (nothing removed) -> saved with grasp_success = false
  u       undo the last pair
  c       tray cleared (every plastic picked): save the final frame, start a new session
  n       new session without clearing
  q       quit

For complete labels, pick EVERY plastic item in a session one at a time, then press c.
Propagation then labels every frame fully, and the cleared frame becomes a hard negative.

    picklabel capture --source http://192.168.1.23:8080/video --out data/pairs
    picklabel import-photos ~/Pictures/tray-run1 --out data/pairs   # alternating before/after photos
    picklabel annotate data/pairs                                   # click grasp points afterwards
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from picklabel.io import IMAGE_EXTS, find_pick_dirs, read_image, read_json, save_pair, write_image, write_json
from picklabel.types import PickPair

WINDOW = "PickLabel rig"


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera source {source!r}")
    return cap


def grab(cap: cv2.VideoCapture, n_avg: int = 1, flush: int = 4) -> np.ndarray:
    """Latest frame (stale buffered frames flushed), averaged over ``n_avg`` reads."""
    for _ in range(flush):
        cap.grab()
    acc, n = None, 0
    for _ in range(max(n_avg, 1) * 3):
        ok, frame = cap.read()
        if not ok:
            continue
        acc = frame.astype(np.float32) if acc is None else acc + frame
        n += 1
        if n >= n_avg:
            break
    if acc is None:
        raise RuntimeError("camera returned no frames")
    return np.clip(acc / n, 0, 255).astype(np.uint8)


class Display:
    """Shows frames scaled to fit the screen and maps clicks back to full resolution."""

    def __init__(self, max_w: int = 1280):
        self.max_w, self.scale, self.click = max_w, 1.0, None
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x / self.scale, y / self.scale)

    def show(self, frame: np.ndarray, lines: list[str], point=None) -> int:
        self.scale = min(1.0, self.max_w / frame.shape[1])
        view = cv2.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale < 1 else frame.copy()
        if point is not None:
            p = (int(point[0] * self.scale), int(point[1] * self.scale))
            cv2.circle(view, p, 10, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(view, p, 10, (60, 255, 60), 2, cv2.LINE_AA)
        for i, text in enumerate(lines):
            cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        return cv2.waitKey(15) & 0xFF


def capture(source: str, out: Path, n_avg: int = 3, class_name: str = "plastic") -> None:
    cap = open_source(source)
    disp = Display()
    session = time.strftime("%Y%m%d-%H%M%S")
    k, before, grasp, saved = 0, None, None, []
    while True:
        frame = grab(cap, 1, flush=0)
        if before is None:
            lines = [f"session {session}  picks {k}", "b: before   c: tray cleared   n: new session   q: quit"]
            key = disp.show(frame, lines)
        elif grasp is None:
            key = disp.show(before, ["click the grasp point on the item"])
            if disp.click is not None:
                grasp, disp.click = disp.click, None
        else:
            key = disp.show(frame, ["remove the item, then a  (m: missed, nothing removed)"], grasp)
        if key == ord("q"):
            break
        if key == ord("b"):
            before, grasp, disp.click = grab(cap, n_avg), None, None
        elif key in (ord("a"), ord("m")) and before is not None and grasp is not None:
            pair = PickPair(pick_id=f"{k:03d}", before=before, after=grab(cap, n_avg), grasp_xy=grasp,
                            session=session, class_name=class_name,
                            grasp_success=False if key == ord("m") else None, timestamp=time.time(),
                            meta={"source": "home-rig", "camera": source})
            saved.append(save_pair(pair, out))
            k, before, grasp = k + 1, None, None
        elif key == ord("u") and saved:
            shutil.rmtree(saved.pop())
            k = max(k - 1, 0)
        elif key in (ord("c"), ord("n")):
            if key == ord("c") and k > 0:
                write_image(out / session / "final_after.jpg", grab(cap, n_avg))
            if k > 0:
                write_json(out / session / "session.json", {"session": session, "cleared": key == ord("c"),
                                                            "final_after": "final_after.jpg" if key == ord("c") else None,
                                                            "n_picks": k})
            session, k, before, grasp, saved = time.strftime("%Y%m%d-%H%M%S"), 0, None, None, []
    cap.release()
    cv2.destroyAllWindows()


def import_photos(src: Path, out: Path, session: str | None = None, class_name: str = "plastic") -> list[Path]:
    """Alternating before/after photos (sorted by name) -> pick folders without grasp points yet.

    If the count is odd, the last photo is taken as the cleared tray.
    """
    photos = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    session = session or src.name
    dirs = []
    for k in range(len(photos) // 2):
        d = out / session / f"{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(photos[2 * k], d / f"before{photos[2 * k].suffix.lower()}")
        shutil.copy2(photos[2 * k + 1], d / f"after{photos[2 * k + 1].suffix.lower()}")
        write_json(d / "pick.json", {"pick_id": f"{k:03d}", "session": session, "class_name": class_name,
                                     "grasp_px": None, "timestamp": float(k), "source": "phone-photos"})
        dirs.append(d)
    if len(photos) % 2 == 1:
        shutil.copy2(photos[-1], out / session / "final_after.jpg")
        write_json(out / session / "session.json", {"session": session, "cleared": True,
                                                    "final_after": "final_after.jpg", "n_picks": len(dirs)})
    return dirs


def annotate(root: Path, redo: bool = False) -> int:
    """Click the grasp point on each before frame that does not have one yet."""
    disp = Display()
    done = 0
    for d in find_pick_dirs(root):
        meta = read_json(d / "pick.json") if (d / "pick.json").exists() else {}
        if meta.get("grasp_px") and not redo:
            continue
        before = read_image(next(d / f"before{e}" for e in IMAGE_EXTS if (d / f"before{e}").exists()))
        disp.click = None
        while disp.click is None:
            key = disp.show(before, [f"{d.parent.name}/{d.name}: click the grasp point", "s: skip   q: quit"])
            if key == ord("q"):
                cv2.destroyAllWindows()
                return done
            if key == ord("s"):
                break
        if disp.click is not None:
            meta["grasp_px"] = [round(disp.click[0], 1), round(disp.click[1], 1)]
            write_json(d / "pick.json", meta)
            done += 1
    cv2.destroyAllWindows()
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Capture before/after pick pairs with grasp points.")
    ap.add_argument("--source", default="0", help="camera index or stream URL")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--avg", type=int, default=3, help="frames averaged per capture")
    ap.add_argument("--class-name", default="plastic")
    args = ap.parse_args(argv)
    capture(args.source, Path(args.out), args.avg, args.class_name)


def main_import(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Import alternating before/after photos as pick folders.")
    ap.add_argument("src")
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--session", default=None)
    args = ap.parse_args(argv)
    dirs = import_photos(Path(args.src), Path(args.out), args.session)
    print(f"imported {len(dirs)} pairs; now run: picklabel annotate {args.out}")


def main_annotate(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Click grasp points for pick folders that lack one.")
    ap.add_argument("root")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)
    print(f"annotated {annotate(Path(args.root), args.redo)} picks")


if __name__ == "__main__":
    main()
