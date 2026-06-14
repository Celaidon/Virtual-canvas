# =============================================================================
# canvas_engine.py — Persistent drawing canvas
#
# UI PASS changes:
#   • update() now interpolates sub-steps between last_point and tip,
#     eliminating gaps that appeared during fast hand movement
#   • Eraser uses cv2.circle (not cv2.line) so the preview circle matches
#     the actual erase area exactly
#   • get_canvas() unchanged — save compatibility preserved
# =============================================================================

import cv2
import numpy as np
import config


class CanvasEngine:

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._canvas     = self._blank_canvas()
        self._last_point: tuple[int, int] | None = None
        self.is_drawing  = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tip           : tuple[int, int] | None,
        drawing_active: bool,
        tool_manager  = None,
    ):
        """
        Called once per frame.  Draws from _last_point → tip with
        sub-step interpolation to fill gaps during fast movement.
        """
        self.is_drawing = drawing_active

        if not drawing_active or tip is None:
            self._last_point = None
            return

        if self._last_point is None:
            self._last_point = tip
            return

        # ---- Interpolated stroke to eliminate gaps ----
        self._draw_segment(self._last_point, tip, tool_manager)
        self._last_point = tip

    def composite(self, webcam_frame: np.ndarray) -> np.ndarray:
        """Overlay strokes onto webcam frame (non-white = drawn)."""
        output = webcam_frame.copy()
        drawn_mask = np.any(self._canvas < 250, axis=2)
        output[drawn_mask] = self._canvas[drawn_mask]
        return output

    def clear(self):
        self._canvas     = self._blank_canvas()
        self._last_point = None
        print("[Canvas] Cleared.")

    def get_canvas(self) -> np.ndarray:
        return self._canvas.copy()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _draw_segment(self, pt_from, pt_to, tool_manager):
        """
        Draw from pt_from → pt_to with enough intermediate steps that
        even fast movement never leaves a visible gap.

        Strategy: compute the pixel distance, then place one stamp every
        (thickness/2) pixels along the segment.  For small movements this
        is just one call; for fast swipes it may be several.
        """
        if tool_manager is None:
            # Fallback — plain pencil
            cv2.line(self._canvas, pt_from, pt_to,
                     config.DEFAULT_COLOR, config.PENCIL_THICKNESS, cv2.LINE_AA)
            return

        from toolbar import ToolName
        profile = tool_manager.get_profile()

        # Eraser uses a circle stamp, not a line, for consistent preview match
        if tool_manager.get_active_tool() == ToolName.ERASER:
            self._stamp_eraser(pt_from, pt_to, tool_manager)
            return

        # All other tools: cv2.line handles anti-aliasing natively,
        # but we add intermediate points to close gaps on fast movement.
        color     = (profile.color if profile.color is not None
                     else tool_manager.current_color)
        thickness = profile.thickness

        dx = pt_to[0] - pt_from[0]
        dy = pt_to[1] - pt_from[1]
        dist = max(1, int((dx*dx + dy*dy) ** 0.5))

        # Step size = half the line thickness so each circle overlaps the last
        step = max(1, thickness // 2)
        steps = max(1, dist // step)

        if steps <= 1:
            # Short move — single line call is fine
            if profile.opacity >= 1.0:
                cv2.line(self._canvas, pt_from, pt_to,
                         color, thickness, profile.line_type)
            else:
                overlay = self._canvas.copy()
                cv2.line(overlay, pt_from, pt_to,
                         color, thickness, profile.line_type)
                cv2.addWeighted(overlay, profile.opacity,
                                self._canvas, 1.0 - profile.opacity,
                                0, self._canvas)
        else:
            # Long / fast move — draw overlapping circles along the path
            # This guarantees continuity regardless of speed.
            if profile.opacity >= 1.0:
                for i in range(steps + 1):
                    t  = i / steps
                    px = int(pt_from[0] + dx * t)
                    py = int(pt_from[1] + dy * t)
                    cv2.circle(self._canvas, (px, py),
                               thickness // 2, color, -1, cv2.LINE_AA)
                # Also draw the capping line for clean edges
                cv2.line(self._canvas, pt_from, pt_to,
                         color, thickness, profile.line_type)
            else:
                overlay = self._canvas.copy()
                for i in range(steps + 1):
                    t  = i / steps
                    px = int(pt_from[0] + dx * t)
                    py = int(pt_from[1] + dy * t)
                    cv2.circle(overlay, (px, py),
                               thickness // 2, color, -1, cv2.LINE_AA)
                cv2.line(overlay, pt_from, pt_to,
                         color, thickness, profile.line_type)
                cv2.addWeighted(overlay, profile.opacity,
                                self._canvas, 1.0 - profile.opacity,
                                0, self._canvas)

    def _stamp_eraser(self, pt_from, pt_to, tool_manager):
        """
        Erase along the path using the dynamic eraser radius from ToolManager.
        Uses filled white circles so preview circle = actual erase area.
        """
        r  = tool_manager.eraser_radius
        dx = pt_to[0] - pt_from[0]
        dy = pt_to[1] - pt_from[1]
        dist = max(1, int((dx*dx + dy*dy) ** 0.5))
        step  = max(1, r // 2)
        steps = max(1, dist // step)

        for i in range(steps + 1):
            t  = i / steps
            px = int(pt_from[0] + dx * t)
            py = int(pt_from[1] + dy * t)
            cv2.circle(self._canvas, (px, py), r,
                       (255, 255, 255), -1)

    def _blank_canvas(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), 255, dtype=np.uint8)