"""Summary: Organism entity with energy, brain, sensing, action selection, and visual categories."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np


def generate_species_color(species_id: int) -> tuple[int, int, int]:
    """
    Generate a distinct color for each species ID.
    Uses HSV color space to ensure good color separation.
    """
    # Predefined colors for first few species (easily distinguishable)
    predefined_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 128), # Gray
        (255, 192, 203), # Pink
    ]
    
    if species_id < len(predefined_colors):
        return predefined_colors[species_id]
    print("error")
    # For species beyond predefined, generate colors using HSV
    # Use golden ratio to get good color distribution
    # golden_ratio = 0.618033988749
    # hue = (species_id * golden_ratio) % 1.0
    
    # # Convert HSV to RGB (with fixed saturation and value for good visibility)
    # saturation = 0.8
    # value = 0.9
    
    # def hsv_to_rgb(h, s, v):
    #     h_i = int(h * 6)
    #     f = h * 6 - h_i
    #     p = v * (1 - s)
    #     q = v * (1 - f * s)
    #     t = v * (1 - (1 - f) * s)
        
    #     if h_i == 0:
    #         r, g, b = v, t, p
    #     elif h_i == 1:
    #         r, g, b = q, v, p
    #     elif h_i == 2:
    #         r, g, b = p, v, t
    #     elif h_i == 3:
    #         r, g, b = p, q, v
    #     elif h_i == 4:
    #         r, g, b = t, p, v
    #     else:
    #         r, g, b = v, p, q
            
    #     return int(r * 255), int(g * 255), int(b * 255)
    
    # return hsv_to_rgb(hue, saturation, value)


@dataclass
class OrganismCategories:
    """Manages organism categories and their visual properties."""
    tags: List[str] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 0, 0)  # RGB color (default red)
    properties: Dict[str, Any] = field(default_factory=dict)  # extensible properties
    
    def add_tag(self, tag: str, duration: int = None):
        """Add a category tag, optionally with expiration."""
        if tag not in self.tags:
            self.tags.append(tag)
        if duration is not None:
            self.properties[f"{tag}_expires"] = duration
    
    def remove_tag(self, tag: str):
        """Remove a category tag."""
        if tag in self.tags:
            self.tags.remove(tag)
        self.properties.pop(f"{tag}_expires", None)
    
    def has_tag(self, tag: str) -> bool:
        """Check if organism has specific tag."""
        return tag in self.tags
    
    def update_tick(self):
        """Update time-based properties and expire tags."""
        expired_tags = []
        for key, value in list(self.properties.items()):
            if key.endswith("_expires") and isinstance(value, int):
                if value <= 1:
                    tag = key.replace("_expires", "")
                    expired_tags.append(tag)
                else:
                    self.properties[key] = value - 1
        
        for tag in expired_tags:
            self.remove_tag(tag)


@dataclass
class Organism:
    id: int
    x: int
    y: int
    energy: float
    brain: any
    species: int = 0  # Species identifier
    categories: OrganismCategories = field(default_factory=OrganismCategories)

    def alive(self) -> bool:
        return self.energy > 0
    
    def tick_update(self):
        """Update organism state each tick (categories, colors, etc.)."""
        self.categories.update_tick()
        self._update_color_based_on_categories()
    
    def _update_color_based_on_categories(self):
        """Update organism color based on current categories and species."""
        # Priority system: special states override species colors
        # if self.categories.has_tag("just_born"):
        #     self.categories.color = (255, 255, 0)  # yellow
        # elif self.categories.has_tag("reproducer"):
        #     self.categories.color = (255, 100, 255)  # magenta
        # elif self.categories.has_tag("high_energy"):
        #     self.categories.color = (255, 255, 255)  # white (high energy)
        # elif self.categories.has_tag("low_energy"):
        #     # Use darker version of species color for low energy
        #     base_color = self._get_species_color()
        #     self.categories.color = tuple(max(50, c // 3) for c in base_color)  # darker
        # else:
        #     # Default to species color
        self.categories.color = self._get_species_color()

    def _get_species_color(self) -> tuple[int, int, int]:
        """Get the color associated with this organism's species."""
        return generate_species_color(self.species)

    def sense(self, grid, sense_radius: int) -> np.ndarray:
        size = grid.size
        feats = []
        for dy in range(-sense_radius, sense_radius+1):
            for dx in range(-sense_radius, sense_radius+1):
                sx, sy = self.x+dx, self.y+dy
                if 0 <= sx < size and 0 <= sy < size:
                    occ = grid.occupancy[sy, sx] != -1
                    food = grid.food[sy, sx] > 0
                else:
                    occ = True
                    food = False
                feats.append(1.0 if occ else 0.0)
                feats.append(-1.0 if food else 0.0)
        feats.append(self.energy)
        feats.append(self.x)
        feats.append(self.y)
        return np.asarray(feats, dtype=np.float32)

    def act(self, logits: np.ndarray, stochastic: bool = False):
        """Select action: argmax (default) or sample from softmax if stochastic."""
        if not stochastic:
            return int(np.argmax(logits))
        # softmax sampling
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        return int(np.random.choice(len(logits), p=probs))
