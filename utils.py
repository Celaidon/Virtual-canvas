# =============================================================================
# utils.py — Shared utility functions used across modules
# =============================================================================

import cv2
import numpy as np


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 1,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    padding: int = 4,
):
    """
    Draw text with a filled rectangle background for readability.

    Parameters
    ----------
    frame      : BGR image to draw onto (modified in place)
    text       : string to display
    pos        : (x, y) top-left of the text baseline
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos

    # Background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def point_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euclidean distance between two (x, y) points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 → a, t=1 → b."""
    return a + (b - a) * t


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between low and high."""
    return max(low, min(high, value))
