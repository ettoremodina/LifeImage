"""Summary: Rendering utilities for frame images and optional live video writing.
Provides:
 - render_occupancy legacy (target overlay)
 - render_entities: organisms with custom colors, food green, background black
 - LiveVideoWriter: append numpy frames directly without saving PNGs
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio


def render_occupancy(occupancy: np.ndarray, target: np.ndarray, out_path: Path, upscale_factor: int = None):
    h, w = occupancy.shape
    occ_mask = (occupancy != -1).astype(np.uint8) * 255
    base = (target * 255).astype(np.uint8)
    dark = (base * 0.2).astype(np.uint8)
    mask3 = np.repeat(occ_mask[:, :, None], 3, axis=2)
    img = np.where(mask3 > 0, base, dark)
    img[occ_mask > 0] = [255, 255, 255]
    
    # auto-calculate upscale factor for small grids
    if upscale_factor is None:
        if max(h, w) < 100:
            upscale_factor = max(1, 800 // max(h, w))  # target ~800px
        else:
            upscale_factor = 1
    
    # upscale if needed using nearest neighbor
    if upscale_factor > 1:
        img = np.repeat(np.repeat(img, upscale_factor, axis=0), upscale_factor, axis=1)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


def render_entities(occupancy: np.ndarray, food: np.ndarray, organisms: list = None, upscale_factor: int = None) -> np.ndarray:
    """Return RGB image: organisms with custom colors, food green, empty black.
    occupancy: int array (-1 empty else organism id)
    food: float array (0 means none)
    organisms: list of organisms with color information
    upscale_factor: if provided, upscale small grids for better visibility
    """
    h, w = occupancy.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # food first (bright green) - make it more visible
    food_mask = food > 0
    img[food_mask] = (0, 255, 0)  # bright green
    
    # organisms with custom colors
    if organisms:
        # create organism id to color mapping
        org_colors = {org.id: org.categories.color for org in organisms}
        
        for y in range(h):
            for x in range(w):
                org_id = occupancy[y, x]
                if org_id != -1 and org_id in org_colors:
                    # if there's food here too, blend colors
                    if food[y, x] > 0:
                        org_color = np.array(org_colors[org_id])
                        food_color = np.array([0, 255, 0])
                        img[y, x] = ((org_color + food_color) // 2).astype(np.uint8)  # blend
                    else:
                        img[y, x] = org_colors[org_id]
    else:
        # fallback to standard red organisms
        org_mask = occupancy != -1
        img[org_mask] = (255, 0, 0)  # bright red
        # mixed cells (org + food) = yellow
        mixed_mask = org_mask & food_mask
        img[mixed_mask] = (255, 255, 0)  # yellow for overlap
    
    # auto-calculate upscale factor for small grids
    if upscale_factor is None:
        if max(h, w) < 100:
            upscale_factor = max(1, 800 // max(h, w))  # target ~800px
        else:
            upscale_factor = 1
    
    # upscale if needed using nearest neighbor
    if upscale_factor > 1:
        img = np.repeat(np.repeat(img, upscale_factor, axis=0), upscale_factor, axis=1)
    
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


def simple_video_from_frames(frames_dir: Path, output_path: Path, fps: int = 8):
    """Simple video creation from PNG frames using ffmpeg."""
    frames = sorted(frames_dir.glob('frame_*.png'))
    if not frames:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for f in frames:
            writer.append_data(imageio.imread(f))
    return True


class CompactVideoWriter:
    """Compact video writer that stores frame data directly without intermediate files."""
    def __init__(self, path: Path, fps: int, grid_size: int):
        self.path = path
        self.fps = fps
        self.grid_size = grid_size
        # calculate upscale factor for small grids
        if grid_size < 100:
            self.upscale_factor = max(1, 800 // grid_size)  # target ~800px
        else:
            self.upscale_factor = 1
        self.frames = []
        
    def append_state(self, occupancy: np.ndarray, food: np.ndarray, organisms: list = None):
        """Store compact frame data: organism positions, food locations, and organism colors."""
        org_positions = np.column_stack(np.where(occupancy != -1))
        food_positions = np.column_stack(np.where(food > 0))
        
        # store organism colors if available
        org_colors = {}
        if organisms:
            for org in organisms:
                if hasattr(org, 'categories'):
                    org_colors[org.id] = org.categories.color
                    
        self.frames.append({
            'orgs': org_positions, 
            'food': food_positions,
            'org_colors': org_colors,
            'occupancy_snapshot': occupancy.copy()  # needed for color mapping
        })
    
    def save_video(self):
        """Convert stored states to video."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(self.path, fps=self.fps) as writer:
            for frame_data in self.frames:
                img = self._render_frame(frame_data)
                writer.append_data(img)
                
    def _render_frame(self, frame_data) -> np.ndarray:
        """Render frame from compact data."""
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # food first (bright green)
        if len(frame_data['food']) > 0:
            img[frame_data['food'][:, 0], frame_data['food'][:, 1]] = (0, 255, 0)
        
        # organisms with custom colors
        if len(frame_data['orgs']) > 0:
            occupancy = frame_data.get('occupancy_snapshot', None)
            org_colors = frame_data.get('org_colors', {})
            
            for y, x in frame_data['orgs']:
                if occupancy is not None:
                    org_id = occupancy[y, x]
                    if org_id in org_colors:
                        # blend with food if present
                        if img[y, x][1] == 255:  # green food present
                            org_color = np.array(org_colors[org_id])
                            food_color = np.array([0, 255, 0])
                            img[y, x] = ((org_color + food_color) // 2).astype(np.uint8)
                        else:
                            img[y, x] = org_colors[org_id]
                    else:
                        img[y, x] = (255, 0, 0)  # default red
                else:
                    img[y, x] = (255, 0, 0)  # fallback red
        
        # upscale if needed using nearest neighbor
        if self.upscale_factor > 1:
            img = np.repeat(np.repeat(img, self.upscale_factor, axis=0), self.upscale_factor, axis=1)
        
        return img
