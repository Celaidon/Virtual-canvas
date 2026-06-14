# =============================================================================
# hand_tracker.py — MediaPipe Tasks API hand-tracking wrapper
#
# MIGRATED from mp.solutions.hands  →  mediapipe.tasks HandLandmarker
# Compatible with MediaPipe 0.10.x + Python 3.12/3.14
#
# Responsibilities:
#   • Download hand_landmarker.task model on first run (cached to disk)
#   • Initialise MediaPipe Tasks HandLandmarker in VIDEO running mode
#   • Process each BGR frame and return structured HandData objects
#   • Draw landmarks / connections onto the frame manually (no legacy utils)
#   • Apply smoothing to fingertip positions to reduce jitter
#
# Public API is IDENTICAL to the old file — no changes needed in main.py:
#   tracker = HandTracker()
#   hands_data = tracker.process_frame(bgr_frame)
#   tracker.draw_landmarks(bgr_frame, hands_data)
#   tracker.release()
# =============================================================================

import os
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

# ---------------------------------------------------------------------------
# Model download — fetched once, cached next to this file
# ---------------------------------------------------------------------------
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


def _ensure_model() -> str:
    """
    Return path to the hand_landmarker.task file.
    Downloads it on first run and caches it locally.
    """
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)

    print("[INFO] hand_landmarker.task not found — downloading (~8 MB)...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[INFO] Model saved to: {_MODEL_PATH}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the MediaPipe hand model.\n"
            f"  URL : {_MODEL_URL}\n"
            f"  Error: {exc}\n\n"
            f"Fix: download the file manually and place it at:\n"
            f"  {_MODEL_PATH}"
        ) from exc

    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe 21-point hand model — unchanged)
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# ---------------------------------------------------------------------------
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# Connections for manual skeleton drawing (pairs of landmark indices)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),(5,17),       # palm cross-connections
]

# Colours for drawing (BGR)
_COL_JOINT = (0, 220, 255)    # yellow-ish dots
_COL_BONE  = (180, 180, 180)  # grey connections
_COL_TIP   = (0, 255, 0)      # green cursor ring


# ---------------------------------------------------------------------------
# HandData — public data container (interface unchanged)
# ---------------------------------------------------------------------------
class HandData:
    """
    Container for one detected hand's processed data.

    Attributes
    ----------
    landmarks_px : list[tuple[int,int]]
        All 21 landmark (x, y) positions in pixel coordinates.
    handedness : str
        "Left" or "Right".
    fingers_up : list[bool]
        [thumb, index, middle, ring, pinky] — True if extended.
    index_tip_smooth : tuple[int,int]
        Smoothed (x, y) of the index fingertip — the main draw cursor.
    palm_center : tuple[int,int]
        Rough palm centre pixel coordinate.
    hand_size : float
        Pixel distance wrist→middle MCP; used to normalise pinch ratios.
    """

    def __init__(self):
        self.landmarks_px     = []
        self.handedness       = "Unknown"
        self.fingers_up       = [False, False, False, False, False]
        self.index_tip_smooth = (0, 0)
        self.thumb_tip        = (0, 0)   # raw thumb tip pixel coord
        self.middle_tip       = (0, 0)   # raw middle fingertip pixel coord
        self.palm_center      = (0, 0)
        self.hand_size        = 1.0
        # Pinch ratios: distance between fingertip pair / hand_size
        # 0.0 = fully pinched, ~0.15+ = wide open
        self.pinch_ratio_thumb_index  = 1.0   # thumb ↔ index
        self.pinch_ratio_thumb_middle = 1.0   # thumb ↔ middle (OK sign)


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------
class HandTracker:
    """
    Wraps the MediaPipe Tasks HandLandmarker and exposes the same API
    that the rest of the project already uses.

    Key differences from the old mp.solutions version
    --------------------------------------------------
    • Uses  RunningMode.VIDEO  instead of the stateful streaming mode.
      Each call to process_frame() passes a monotonically increasing
      timestamp (ms) so MediaPipe can apply its internal tracking filter.
    • The Tasks API returns NormalizedLandmark objects (each has .x .y .z)
      instead of the old proto-based landmark list — conversion is handled
      internally; callers still receive plain (int, int) pixel tuples.
    • Handedness: Tasks API returns "Left"/"Right" per the *camera* view
      (i.e. already flipped for a mirrored webcam feed), matching the
      old behaviour.
    • draw_landmarks() no longer delegates to mp.solutions.drawing_utils
      (which is gone); it draws manually with cv2 instead.
    """

    def __init__(self):
        model_path = _ensure_model()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,   # best for webcam loops
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.TRACKING_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Monotonic timestamp counter (milliseconds) required by VIDEO mode
        self._start_ms   = int(time.monotonic() * 1000)

        # Smoothing buffers — one deque per hand slot
        self._smooth_buffers = [
            deque(maxlen=config.SMOOTHING_WINDOW)
            for _ in range(config.MAX_HANDS)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> list[HandData]:
        """
        Detect and process hands in *bgr_frame*.

        Returns a list of HandData (0–MAX_HANDS items).
        The frame is NOT modified here; call draw_landmarks() separately.
        """
        h, w = bgr_frame.shape[:2]

        # Tasks API needs an mp.Image in RGB format
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonically increasing timestamp (ms) — required in VIDEO mode
        timestamp_ms = int(time.monotonic() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_out: list[HandData] = []

        if not result.hand_landmarks:
            # No hands — clear smoothing buffers so old data doesn't persist
            for buf in self._smooth_buffers:
                buf.clear()
            return hands_out

        for hand_idx, (norm_lms, handedness_list) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            if hand_idx >= config.MAX_HANDS:
                break

            data = HandData()

            # Handedness label: Tasks API gives a Category object
            data.handedness = handedness_list[0].display_name  # "Left" or "Right"

            # Convert normalised landmarks → pixel coordinates
            data.landmarks_px = [
                (int(lm.x * w), int(lm.y * h))
                for lm in norm_lms
            ]

            # Derived measurements
            data.palm_center = self._calc_palm_center(data.landmarks_px)
            data.hand_size   = self._calc_hand_size(data.landmarks_px)

            # Finger extension state
            data.fingers_up = self._fingers_up(data.landmarks_px)

            # Smoothed index-tip cursor
            raw_tip = data.landmarks_px[INDEX_TIP]
            data.index_tip_smooth = self._smooth_tip(hand_idx, raw_tip)

            # Convenience tip references
            data.thumb_tip  = data.landmarks_px[THUMB_TIP]
            data.middle_tip = data.landmarks_px[MIDDLE_TIP]

            # Pinch ratios (normalised by hand_size so depth doesn't matter)
            data.pinch_ratio_thumb_index  = self._pinch_ratio(
                data.landmarks_px, THUMB_TIP, INDEX_TIP, data.hand_size)
            data.pinch_ratio_thumb_middle = self._pinch_ratio(
                data.landmarks_px, THUMB_TIP, MIDDLE_TIP, data.hand_size)

            # Keep the raw normalised landmarks for draw_landmarks()
            data._norm_lms = norm_lms   # private

            hands_out.append(data)

        return hands_out

    def draw_landmarks(self, bgr_frame: np.ndarray, hands_data: list[HandData]):
        """
        Draw skeleton + landmark dots onto *bgr_frame* in place.

        Uses plain cv2 calls instead of the removed mp.solutions.drawing_utils.
        Also draws a green ring at the smoothed index-tip cursor.
        """
        if not config.SHOW_LANDMARKS:
            return

        for data in hands_data:
            lms = data.landmarks_px   # list of (x, y) pixel tuples

            # ---- Bone connections ----
            for (a, b) in _HAND_CONNECTIONS:
                cv2.line(bgr_frame, lms[a], lms[b], _COL_BONE, 2, cv2.LINE_AA)

            # ---- Joint dots ----
            for pt in lms:
                cv2.circle(bgr_frame, pt, 5, _COL_JOINT, -1, cv2.LINE_AA)

            # ---- Smoothed cursor ring on index tip ----
            cx, cy = data.index_tip_smooth
            cv2.circle(bgr_frame, (cx, cy), 10, _COL_TIP, 2,  cv2.LINE_AA)
            cv2.circle(bgr_frame, (cx, cy), 3,  _COL_TIP, -1, cv2.LINE_AA)

    def release(self):
        """Release MediaPipe resources. Call when the app closes."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers (logic identical to original — only input format
    # changed: was mp proto objects, now plain (int,int) tuples which
    # the old helpers already expected after the conversion step above)
    # ------------------------------------------------------------------

    def _fingers_up(self, lms: list[tuple[int, int]]) -> list[bool]:
        """
        Return [thumb_up, index_up, middle_up, ring_up, pinky_up].

        • Thumb : tip X vs IP X relative to MCP X direction.
        • Others: tip Y < PIP Y  →  finger is above second knuckle → extended.
          (Y increases downward in image space)
        """
        up = [False] * 5

        # Thumb
        tip_x = lms[THUMB_TIP][0]
        ip_x  = lms[THUMB_IP][0]
        mcp_x = lms[THUMB_MCP][0]
        if tip_x < mcp_x:
            up[0] = tip_x < ip_x
        else:
            up[0] = tip_x > ip_x

        # Four fingers
        for i, (tip_idx, pip_idx) in enumerate([
            (INDEX_TIP,  INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP,   RING_PIP),
            (PINKY_TIP,  PINKY_PIP),
        ]):
            up[i + 1] = lms[tip_idx][1] < lms[pip_idx][1]

        return up

    def _smooth_tip(self, hand_idx: int, raw: tuple[int, int]) -> tuple[int, int]:
        """
        Rolling-average smoothing with a dead-zone to suppress micro-tremor.
        """
        buf = self._smooth_buffers[hand_idx]

        if buf:
            last = buf[-1]
            dist = ((raw[0] - last[0]) ** 2 + (raw[1] - last[1]) ** 2) ** 0.5
            if dist < config.MIN_MOVE_THRESHOLD:
                xs = [p[0] for p in buf]
                ys = [p[1] for p in buf]
                return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

        buf.append(raw)
        xs = [p[0] for p in buf]
        ys = [p[1] for p in buf]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def _pinch_ratio(self, lms, tip_a, tip_b, hand_size):
        ax, ay = lms[tip_a]
        bx, by = lms[tip_b]
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return dist / hand_size

    def _calc_palm_center(self, lms: list[tuple[int, int]]) -> tuple[int, int]:
        key = [lms[WRIST], lms[INDEX_MCP], lms[MIDDLE_MCP], lms[RING_MCP], lms[PINKY_MCP]]
        return (int(sum(p[0] for p in key) / len(key)),
                int(sum(p[1] for p in key) / len(key)))

    def _calc_hand_size(self, lms: list[tuple[int, int]]) -> float:
        wx, wy = lms[WRIST]
        mx, my = lms[MIDDLE_MCP]
        return max(((mx - wx) ** 2 + (my - wy) ** 2) ** 0.5, 1.0)