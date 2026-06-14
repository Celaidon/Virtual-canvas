# =============================================================================
# gesture_detector.py — Gesture recognition layer
#
# STEP 4: Interpret HandData finger states into stable, debounced gestures.
#
# Responsibilities:
#   • Raw gesture classification from a single HandData snapshot
#   • Frame-level debouncing so gestures must be held before activating
#   • Expose a clean GestureResult object to callers
#   • Visual debug overlay (gesture name + confidence bar on screen)
#
# NOT handled here:
#   • Tool switching behaviour        (Step 5)
#   • Eraser / zoom / save logic      (Steps 6-9)
#   • Any canvas or UI mutation
#
# Usage:
#   detector = GestureDetector()
#   result   = detector.update(hand_data)   # call once per frame per hand
#   print(result.name, result.is_active)
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque
from enum import Enum

import config


# ---------------------------------------------------------------------------
# Gesture names — single source of truth used everywhere
# ---------------------------------------------------------------------------

class GestureName(str, Enum):
    """
    All recognisable gesture tokens.
    Using (str, Enum) lets you compare with plain strings and print cleanly.
    """
    NONE         = "NONE"
    DRAW         = "DRAW"          # index only up
    BRUSH        = "BRUSH"         # index + middle up
    THREE_FINGER = "THREE FINGER"  # index + middle + ring up
    OPEN_PALM    = "OPEN PALM"     # all fingers up
    FIST         = "FIST"          # no fingers up
    PINCH        = "PINCH"         # thumb + index close together
    OK_SIGN      = "OK SIGN"       # thumb + middle close, ring+pinky down


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    """
    The stable, debounced gesture reported to the rest of the application.

    Attributes
    ----------
    name        : GestureName  -- which gesture is committed/active
    is_active   : bool         -- True once debounce window is filled
    confidence  : float [0,1]  -- classifier confidence (1.0 for binary gestures)
    raw_name    : GestureName  -- instantaneous (non-debounced) label
    hold_frames : int          -- consecutive frames the raw gesture has been stable
    """
    name        : GestureName = GestureName.NONE
    is_active   : bool        = False
    confidence  : float       = 0.0
    raw_name    : GestureName = GestureName.NONE
    hold_frames : int         = 0


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Per-hand gesture detector with debouncing.

    One instance serves the primary hand. For two-hand gestures (Step 7+),
    create a second instance or extend update() to accept two HandData objects.

    Parameters
    ----------
    debounce_frames : int | None
        Consecutive stable frames before a gesture is committed.
        Defaults to config.GESTURE_DEBOUNCE_FRAMES.
    """

    def __init__(self, debounce_frames: int | None = None):
        self._debounce_target: int = (
            debounce_frames
            if debounce_frames is not None
            else config.GESTURE_DEBOUNCE_FRAMES
        )

        # Sliding window of recent raw classifications
        self._raw_history: deque[GestureName] = deque(
            maxlen=self._debounce_target
        )
        self._committed_gesture = GestureName.NONE
        self._hold_frames       = 0
        self._last_result       = GestureResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, hand) -> GestureResult:
        """
        Classify the gesture in *hand* (HandData) and return a debounced
        GestureResult. Pass hand=None when no hand is detected.
        """
        if hand is None:
            return self._reset()

        # --- 1. Raw instantaneous classification ---
        raw, confidence = self._classify(hand)

        # --- 2. Track consecutive hold frames ---
        if self._raw_history and raw != self._raw_history[-1]:
            # Gesture changed mid-window — restart counter
            self._hold_frames = 1
        else:
            self._hold_frames = min(self._hold_frames + 1, self._debounce_target)

        self._raw_history.append(raw)

        # --- 3. Commit when the whole window agrees ---
        if (len(self._raw_history) == self._debounce_target
                and len(set(self._raw_history)) == 1):
            self._committed_gesture = raw

        is_active = (self._committed_gesture == raw
                     and raw != GestureName.NONE)

        result = GestureResult(
            name        = self._committed_gesture,
            is_active   = is_active,
            confidence  = confidence,
            raw_name    = raw,
            hold_frames = self._hold_frames,
        )
        self._last_result = result
        return result

    def reset(self):
        """Manually reset debounce state."""
        self._reset()

    # ------------------------------------------------------------------
    # Individual gesture detectors (public so callers can use them too)
    # Each returns (is_detected: bool, confidence: float)
    # ------------------------------------------------------------------

    def detect_draw_gesture(self, hand) -> tuple[bool, float]:
        """DRAW — only index finger extended (thumb ignored)."""
        _, index, middle, ring, pinky = hand.fingers_up
        active = index and not middle and not ring and not pinky
        return active, 1.0 if active else 0.0

    def detect_brush_gesture(self, hand) -> tuple[bool, float]:
        """BRUSH — index + middle up, ring + pinky down."""
        _, index, middle, ring, pinky = hand.fingers_up
        active = index and middle and not ring and not pinky
        return active, 1.0 if active else 0.0

    def detect_three_finger(self, hand) -> tuple[bool, float]:
        """THREE FINGER — index + middle + ring up, pinky down."""
        _, index, middle, ring, pinky = hand.fingers_up
        active = index and middle and ring and not pinky
        return active, 1.0 if active else 0.0

    def detect_open_palm(self, hand) -> tuple[bool, float]:
        """OPEN PALM — all four fingers extended."""
        _, index, middle, ring, pinky = hand.fingers_up
        active = index and middle and ring and pinky
        return active, 1.0 if active else 0.0

    def detect_fist(self, hand) -> tuple[bool, float]:
        """FIST — no fingers extended (thumb ignored)."""
        _, index, middle, ring, pinky = hand.fingers_up
        active = not index and not middle and not ring and not pinky
        return active, 1.0 if active else 0.0

    def detect_pinch(self, hand) -> tuple[bool, float]:
        """
        PINCH — thumb tip near index tip.
        Uses pre-computed pinch_ratio_thumb_index (normalised by hand_size).
        Confidence = 1.0 when fully closed, 0.0 at PINCH_OPEN_RATIO.
        """
        ratio  = hand.pinch_ratio_thumb_index
        active = ratio < config.PINCH_CLOSE_RATIO
        span   = max(config.PINCH_OPEN_RATIO - config.PINCH_CLOSE_RATIO, 1e-6)
        confidence = max(0.0, min(1.0,
            (config.PINCH_OPEN_RATIO - ratio) / span
        ))
        return active, confidence

    def detect_ok_sign(self, hand) -> tuple[bool, float]:
        """
        OK SIGN — thumb tip near middle tip + ring & pinky down.
        Uses pre-computed pinch_ratio_thumb_middle.
        """
        _, _, _, ring, pinky = hand.fingers_up
        ratio  = hand.pinch_ratio_thumb_middle
        active = ratio < config.OK_PINCH_RATIO and not ring and not pinky
        confidence = max(0.0, min(1.0,
            (config.OK_PINCH_RATIO - ratio) / max(config.OK_PINCH_RATIO, 1e-6)
        )) if active else 0.0
        return active, confidence

    # ------------------------------------------------------------------
    # Visual debug overlay
    # ------------------------------------------------------------------

    def draw_debug(self, frame: np.ndarray, result: GestureResult,
                   pos: tuple[int, int] = (10, 70)):
        """
        Render a gesture debug panel onto *frame* in place.

        Panel shows:
          - Committed gesture name (colour-coded)
          - Raw/instantaneous gesture if it differs
          - Debounce hold-progress bar
          - Confidence bar
        """
        x, y = pos

        # Colour per gesture
        COLOR_MAP: dict[GestureName, tuple] = {
            GestureName.DRAW        : (0,   220, 0  ),
            GestureName.BRUSH       : (0,   165, 255),
            GestureName.THREE_FINGER: (0,   220, 220),
            GestureName.OPEN_PALM   : (80,  80,  255),
            GestureName.FIST        : (0,   0,   220),
            GestureName.PINCH       : (200, 0,   200),
            GestureName.OK_SIGN     : (0,   200, 180),
            GestureName.NONE        : (120, 120, 120),
        }
        color = COLOR_MAP.get(result.name, (120, 120, 120))

        # Semi-transparent background panel
        panel_w, panel_h = 270, 95
        roi = frame[max(0, y - 8) : y + panel_h,
                    max(0, x - 8) : x + panel_w + 8]
        if roi.size > 0:
            dark = np.zeros_like(roi)
            cv2.addWeighted(dark, 0.50, roi, 0.50, 0, roi)
            frame[max(0, y - 8) : y + panel_h,
                  max(0, x - 8) : x + panel_w + 8] = roi

        # Committed gesture label
        cv2.putText(frame, f"Gesture: {result.name}",
                    (x, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2, cv2.LINE_AA)

        # Raw label (shown only when different from committed)
        if result.raw_name != result.name:
            cv2.putText(frame, f"  raw: {result.raw_name}",
                        (x, y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (160, 160, 160), 1, cv2.LINE_AA)

        # --- Debounce hold bar ---
        bar_y  = y + 54
        bar_w  = 200
        bar_h  = 10
        fill   = int(bar_w * result.hold_frames / max(self._debounce_target, 1))
        fill   = min(fill, bar_w)
        cv2.rectangle(frame, (x, bar_y),
                      (x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
        bar_color = color if result.is_active else (90, 90, 90)
        if fill > 0:
            cv2.rectangle(frame, (x, bar_y),
                          (x + fill, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, "hold", (x + bar_w + 6, bar_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        # --- Confidence bar ---
        conf_y = bar_y + 16
        conf_w = int(bar_w * result.confidence)
        cv2.rectangle(frame, (x, conf_y),
                      (x + bar_w, conf_y + bar_h), (55, 55, 55), -1)
        if conf_w > 0:
            cv2.rectangle(frame, (x, conf_y),
                          (x + conf_w, conf_y + bar_h), (0, 180, 255), -1)
        cv2.putText(frame, "conf", (x + bar_w + 6, conf_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, hand) -> tuple[GestureName, float]:
        """
        Single-frame raw classification. Priority order is intentional:
        pinch/ok checked before finger-count gestures because a pinch
        can coexist with partially-raised fingers that would otherwise
        match DRAW or BRUSH.
        """
        ok, ok_conf = self.detect_ok_sign(hand)
        if ok:
            return GestureName.OK_SIGN, ok_conf

        pinch, pinch_conf = self.detect_pinch(hand)
        if pinch:
            return GestureName.PINCH, pinch_conf

        palm, _ = self.detect_open_palm(hand)
        if palm:
            return GestureName.OPEN_PALM, 1.0

        three, _ = self.detect_three_finger(hand)
        if three:
            return GestureName.THREE_FINGER, 1.0

        brush, _ = self.detect_brush_gesture(hand)
        if brush:
            return GestureName.BRUSH, 1.0

        draw, _ = self.detect_draw_gesture(hand)
        if draw:
            return GestureName.DRAW, 1.0

        fist, _ = self.detect_fist(hand)
        if fist:
            return GestureName.FIST, 1.0

        return GestureName.NONE, 0.0

    def _reset(self) -> GestureResult:
        """Clear debounce state and return an empty result."""
        self._raw_history.clear()
        self._committed_gesture = GestureName.NONE
        self._hold_frames       = 0
        result = GestureResult()
        self._last_result = result
        return result