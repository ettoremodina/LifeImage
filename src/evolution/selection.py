"""Summary: Selection utilities (placeholder strict energy-based culling)."""
from __future__ import annotations


def cull_dead(organisms):
    return [o for o in organisms if o.alive()]
