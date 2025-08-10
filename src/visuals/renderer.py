"""Summary: Rendering utilities for frame images and optional live video writing.
Provides:
 - render_occupancy legacy (target overlay)
 - render_entities: organisms (red), food (green), background black
 - LiveVideoWriter: append numpy frames directly without saving PNGs
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import glob



def render_occupancy(occupancy: np.ndarray, target: np.ndarray, out_path: Path):
    h, w = occupancy.shape
    occ_mask = (occupancy != -1).astype(np.uint8) * 255
    base = (target * 255).astype(np.uint8)
    dark = (base * 0.2).astype(np.uint8)
    mask3 = np.repeat(occ_mask[:, :, None], 3, axis=2)
    img = np.where(mask3 > 0, base, dark)
    img[occ_mask > 0] = [255, 255, 255]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


def render_entities(occupancy: np.ndarray, food: np.ndarray) -> np.ndarray:
    """Return RGB image: organisms red, food green, empty black.
    occupancy: int array (-1 empty else organism id)
    food: float array (0 means none)
    """
    h, w = occupancy.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # food first (green)
    food_mask = food > 0
    img[food_mask] = (0, 160, 0)
    # organisms overwrite (red)
    org_mask = occupancy != -1
    img[org_mask] = (200, 0, 0)
    return img


class LiveVideoWriter:
    """Write video frames directly (numpy arrays) skipping disk PNGs.
    Uses imageio writer. Frames must be consistent size.
    """
    def __init__(self, path: Path, fps: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        if ext == '.mp4':
            try:
                import imageio_ffmpeg  # noqa: F401
            except Exception:
                raise RuntimeError("ffmpeg backend missing. Install with: pip install imageio[ffmpeg]")
            self._writer = imageio.get_writer(path, fps=fps)
        else:
            self._writer = imageio.get_writer(path, mode='I', fps=fps)
        self.closed = False

    def append(self, frame: np.ndarray):
        if self.closed:
            return
        self._writer.append_data(frame)

    def close(self):
        if not self.closed:
            try:
                self._writer.close()
            finally:
                self.closed = True


def frames_to_video(frames_dir: Path, pattern: str, output_path: Path, fps: int = 8, fallback_gif: bool = True, delete_frames: bool = False):
    """Create a video (mp4 or gif) from frame images with diagnostics and fallback.
    pattern: glob pattern relative to frames_dir (e.g., 'frame_*.png')
    Returns True if file written else False.
    """
    print(f"[frames_to_video] start frames_dir={frames_dir} pattern={pattern} output={output_path}", flush=True)
    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        print(f"[frames_to_video] no frames found: {frames_dir / pattern}", flush=True)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        ext = output_path.suffix.lower()
        if ext == '.mp4':
            try:
                import imageio_ffmpeg  # noqa: F401
            except Exception:
                raise RuntimeError("ffmpeg backend missing. Install with: pip install imageio[ffmpeg]")
            writer = imageio.get_writer(output_path, fps=fps)
        else:
            writer = imageio.get_writer(output_path, mode='I', fps=fps)
        print(f"[frames_to_video] writing {len(frames)} frames to {output_path}", flush=True)
        for f in frames:
            img = imageio.imread(f)
            writer.append_data(img)
        writer.close()
        print(f"[frames_to_video] success saved {output_path}", flush=True)
        if delete_frames:
            removed = 0
            for f in frames:
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass
            print(f"[frames_to_video] deleted {removed} frame files", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[frames_to_video] error: {e}", flush=True)
        traceback.print_exc()
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if fallback_gif:
            gif_path = output_path.with_suffix('.gif')
            try:
                print(f"[frames_to_video] attempting GIF fallback -> {gif_path}", flush=True)
                with imageio.get_writer(gif_path, mode='I', duration=1.0/max(1,fps)) as gifw:
                    for f in frames:
                        gifw.append_data(imageio.imread(f))
                print(f"[frames_to_video] GIF saved to {gif_path}", flush=True)
                if delete_frames:
                    removed = 0
                    for f in frames:
                        try:
                            f.unlink()
                            removed += 1
                        except Exception:
                            pass
                    print(f"[frames_to_video] deleted {removed} frame files", flush=True)
                return True
            except Exception as ge:
                print(f"[frames_to_video] GIF fallback failed: {ge}", flush=True)
        return False
