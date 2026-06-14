# =============================================================================
# toolbar.py — Tool manager + all HUD panel rendering
#
# FRAME REDESIGN:
#   • All panels now render into the LEFT or RIGHT regions of the 1920x1080
#     output frame (not overlaid on the webcam area)
#   • Pastel kawaii visual style matching the decorative border
#   • draw_left_panel()  → tool info, color, size in the left artwork zone
#   • draw_right_panel() → shortcuts in the right artwork zone
#   • draw_palette()     → color swatches inside left panel (picker mode)
#   • draw_cursor()      → unchanged tool-aware cursor (drawn on webcam layer)
#   • draw_with_tool()   → unchanged drawing logic
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum

import config


class ToolName(str, Enum):
    PENCIL       = "pencil"
    BRUSH        = "brush"
    INK_PEN      = "ink_pen"
    ERASER       = "eraser"
    COLOR_PICKER = "color_picker"


@dataclass
class ToolProfile:
    name         : ToolName
    thickness    : int
    color        : tuple | None
    opacity      : float
    line_type    : int
    label        : str
    cursor_color : tuple
    cursor_radius: int


# ---------------------------------------------------------------------------
# Pastel card drawing helpers
# ---------------------------------------------------------------------------

def _pastel_card(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 alpha: float = 0.72, radius: int = 14):
    """Draw a rounded-corner semi-transparent pastel card."""
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    roi = frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return
    card_color = np.array(config.PASTEL_CARD_BG, dtype=np.uint8)
    card = np.full_like(roi, card_color)
    cv2.addWeighted(card, alpha, roi, 1 - alpha, 0, roi)
    frame[y1c:y2c, x1c:x2c] = roi
    # Border
    cv2.rectangle(frame, (x1c, y1c), (x2c-1, y2c-1),
                  config.PASTEL_BORDER, 1, cv2.LINE_AA)


def _label(frame, text, x, y, scale=0.48, color=None, bold=False):
    color = color or config.PASTEL_PINK_DARK
    thick = 2 if bold else 1
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _divider(frame, x1, x2, y):
    cv2.line(frame, (x1, y), (x2, y), config.PASTEL_BORDER, 1)


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class ToolManager:

    _PROFILES: dict[ToolName, ToolProfile] = {
        ToolName.PENCIL: ToolProfile(
            name=ToolName.PENCIL, thickness=config.PENCIL_THICKNESS,
            color=None, opacity=1.0, line_type=cv2.LINE_AA,
            label="Pencil", cursor_color=(120, 200, 120), cursor_radius=6,
        ),
        ToolName.BRUSH: ToolProfile(
            name=ToolName.BRUSH, thickness=config.BRUSH_THICKNESS,
            color=None, opacity=config.BRUSH_OPACITY, line_type=cv2.LINE_AA,
            label="Brush", cursor_color=(120, 170, 240),
            cursor_radius=config.BRUSH_THICKNESS // 2 + 4,
        ),
        ToolName.INK_PEN: ToolProfile(
            name=ToolName.INK_PEN, thickness=config.INKPEN_THICKNESS,
            color=config.INKPEN_COLOR, opacity=config.INKPEN_OPACITY,
            line_type=cv2.LINE_AA,
            label="Ink Pen", cursor_color=(100, 100, 100), cursor_radius=6,
        ),
        ToolName.ERASER: ToolProfile(
            name=ToolName.ERASER, thickness=config.ERASER_THICKNESS_MIN,
            color=(255, 255, 255), opacity=1.0, line_type=cv2.LINE_8,
            label="Eraser", cursor_color=(160, 120, 200), cursor_radius=20,
        ),
        ToolName.COLOR_PICKER: ToolProfile(
            name=ToolName.COLOR_PICKER, thickness=1,
            color=None, opacity=1.0, line_type=cv2.LINE_AA,
            label="Color Picker", cursor_color=(100, 200, 180), cursor_radius=14,
        ),
    }

    def __init__(self,
                 initial_tool : ToolName | str = ToolName.PENCIL,
                 initial_color: tuple           = config.DEFAULT_COLOR):
        self._active       : ToolName = ToolName(initial_tool)
        self._prev_tool    : ToolName = ToolName(initial_tool)
        self.current_color : tuple    = initial_color
        self._pending_tool  : ToolName | None = None
        self._pending_frames: int             = 0
        self.eraser_radius  : int             = config.ERASER_THICKNESS_MIN // 2
        # Color picker state
        self._hover_swatch : int | None = None
        self._hover_frames : int        = 0
        self._palette_rects: list       = []

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    def get_active_tool(self) -> ToolName:
        return self._active

    def get_profile(self, tool: ToolName | None = None) -> ToolProfile:
        return self._PROFILES[tool or self._active]

    def set_active_tool(self, tool: ToolName | str, immediate: bool = False):
        tool = ToolName(tool)
        if tool == self._active:
            self._pending_tool   = None
            self._pending_frames = 0
            return
        if immediate:
            self._commit_tool(tool)
            return
        if tool == self._pending_tool:
            self._pending_frames += 1
            if self._pending_frames >= config.TOOL_SWITCH_DEBOUNCE_FRAMES:
                self._commit_tool(tool)
        else:
            self._pending_tool   = tool
            self._pending_frames = 1

    def update_from_gesture(self, gesture_name: str):
        from gesture_detector import GestureName
        MAP = {
            GestureName.DRAW        : ToolName.PENCIL,
            GestureName.BRUSH       : ToolName.BRUSH,
            GestureName.OPEN_PALM   : ToolName.ERASER,
            GestureName.THREE_FINGER: ToolName.COLOR_PICKER,
        }
        target = MAP.get(gesture_name)
        if target:
            self.set_active_tool(target, immediate=False)

    def update_eraser_size(self, hand):
        if hand is None:
            return
        raw_r = int(hand.hand_size * config.ERASER_HAND_SCALE * 0.5)
        clamped = max(config.ERASER_THICKNESS_MIN // 2,
                      min(config.ERASER_THICKNESS_MAX // 2, raw_r))
        self.eraser_radius = int(self.eraser_radius * 0.7 + clamped * 0.3)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_with_tool(self, canvas: np.ndarray,
                       pt_from: tuple, pt_to: tuple):
        profile   = self._PROFILES[self._active]
        color     = profile.color if profile.color is not None else self.current_color
        thickness = profile.thickness
        if profile.opacity >= 1.0:
            cv2.line(canvas, pt_from, pt_to, color, thickness, profile.line_type)
        else:
            overlay = canvas.copy()
            cv2.line(overlay, pt_from, pt_to, color, thickness, profile.line_type)
            cv2.addWeighted(overlay, profile.opacity,
                            canvas, 1.0 - profile.opacity, 0, canvas)

    # ------------------------------------------------------------------
    # Color picker
    # ------------------------------------------------------------------

    def update_color_picker(self, tip: tuple | None) -> bool:
        if tip is None or not self._palette_rects:
            self._hover_swatch = None
            self._hover_frames = 0
            return False
        tx, ty = tip
        hit = None
        for i, (x1, y1, x2, y2) in enumerate(self._palette_rects):
            if x1 <= tx <= x2 and y1 <= ty <= y2:
                hit = i
                break
        if hit is None:
            self._hover_swatch = None
            self._hover_frames = 0
            return False
        if hit == self._hover_swatch:
            self._hover_frames += 1
        else:
            self._hover_swatch = hit
            self._hover_frames = 1
        if self._hover_frames >= config.COLOR_HOVER_FRAMES:
            self.current_color = config.PALETTE_COLORS[hit]
            self._hover_swatch = None
            self._hover_frames = 0
            if config.COLOR_PICKER_AUTO_RETURN:
                self._commit_tool(self._prev_tool)
            return True
        return False

    # ------------------------------------------------------------------
    # Cursor (drawn on the webcam layer, in webcam-local coordinates)
    # ------------------------------------------------------------------

    def draw_cursor(self, frame: np.ndarray,
                    tip: tuple | None,
                    is_drawing: bool,
                    is_paused: bool = False):
        if tip is None:
            return
        if is_paused:
            cx, cy = tip
            s = 12
            cv2.line(frame,(cx-s,cy-s),(cx+s,cy+s),(130,130,130),2,cv2.LINE_AA)
            cv2.line(frame,(cx+s,cy-s),(cx-s,cy+s),(130,130,130),2,cv2.LINE_AA)
            return

        tool = self._active
        if tool == ToolName.ERASER:
            r = self.eraser_radius
            cv2.circle(frame, tip, r, (160,120,200), 2, cv2.LINE_AA)
            cv2.circle(frame, tip, max(2,r//4), (160,120,200), -1, cv2.LINE_AA)
        elif tool == ToolName.COLOR_PICKER:
            cv2.circle(frame, tip, 16, self.current_color, -1, cv2.LINE_AA)
            cv2.circle(frame, tip, 18, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(frame, tip, 20, (100,200,180), 2, cv2.LINE_AA)
        elif tool == ToolName.PENCIL:
            cv2.circle(frame, tip, 5,  (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(frame, tip, 2,  self.current_color, -1, cv2.LINE_AA)
        elif tool == ToolName.BRUSH:
            r = self._PROFILES[ToolName.BRUSH].cursor_radius
            cv2.circle(frame, tip, r,   self.current_color, -1, cv2.LINE_AA)
            cv2.circle(frame, tip, r+2, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(frame, tip, r+4, (120,170,240), 1, cv2.LINE_AA)
        elif tool == ToolName.INK_PEN:
            cv2.circle(frame, tip, 6,  (40,40,40),  -1, cv2.LINE_AA)
            cv2.circle(frame, tip, 8,  (160,160,160), 1, cv2.LINE_AA)
        else:
            p = self._PROFILES[tool]
            c = p.cursor_color
            cv2.circle(frame, tip, p.cursor_radius, c, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # LEFT PANEL — tool info (drawn on the full 1920x1080 output frame)
    # ------------------------------------------------------------------

    def draw_left_panel(self, frame: np.ndarray, is_paused: bool = False,
                        gesture_label: str = ""):
        if not config.SHOW_TOOL_HUD:
            return

        x1 = config.LEFT_PANEL_X1 + 4
        x2 = config.LEFT_PANEL_X2 - 4
        y1 = config.LEFT_PANEL_Y1 + 4
        y2 = config.LEFT_PANEL_Y2 - 4
        cx = (x1 + x2) // 2    # center x of panel
        pw = x2 - x1

        profile = self._PROFILES[self._active]
        tool_key = self._active.value
        accent   = tuple(config.TOOL_COLORS.get(tool_key, config.PASTEL_ACCENT))

        # ---- Main card background ----
        _pastel_card(frame, x1, y1, x2, y2, alpha=0.78)

        # ---- Header: "TOOL" label ----
        hy = y1 + 28
        _label(frame, "ACTIVE TOOL", x1 + 16, hy,
               scale=0.46, color=config.PASTEL_PINK_MID)

        # ---- Tool name ----
        tool_label = "PAUSED" if is_paused else profile.label.upper()
        tcolor     = (100, 100, 100) if is_paused else accent
        cv2.putText(frame, tool_label, (x1 + 14, hy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, tcolor, 2, cv2.LINE_AA)

        _divider(frame, x1 + 10, x2 - 10, hy + 44)

        # ---- Color swatch ----
        sy     = hy + 60
        sw, sh = 36, 36
        sx     = x1 + 14
        draw_color = (profile.color if profile.color is not None
                      else self.current_color)
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), draw_color, -1)
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), config.PASTEL_BORDER, 1)
        _label(frame, "Color", sx + sw + 10, sy + 14,
               scale=0.44, color=config.PASTEL_PINK_MID)
        # hex-style display
        b, g, r = draw_color
        hex_str = f"#{r:02X}{g:02X}{b:02X}"
        _label(frame, hex_str, sx + sw + 10, sy + 30,
               scale=0.38, color=config.PASTEL_PINK_DARK)

        _divider(frame, x1+10, x2-10, sy + sh + 14)

        # ---- Brush size ----
        bsy = sy + sh + 30
        size_val = (self.eraser_radius * 2
                    if self._active == ToolName.ERASER
                    else profile.thickness)
        _label(frame, "Brush Size", x1+14, bsy,
               scale=0.44, color=config.PASTEL_PINK_MID)
        _label(frame, f"{size_val} px", x1+14, bsy+22,
               scale=0.60, color=accent, bold=True)

        # Size bar
        bar_x, bar_y = x1+14, bsy+32
        bar_w = pw - 28
        bar_h = 8
        max_t = config.ERASER_THICKNESS_MAX
        fill  = int(bar_w * min(size_val / max_t, 1.0))
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),
                      config.PASTEL_BORDER, -1)
        if fill > 0:
            cv2.rectangle(frame,(bar_x,bar_y),(bar_x+fill,bar_y+bar_h),
                          accent, -1)

        _divider(frame, x1+10, x2-10, bar_y+bar_h+14)

        # ---- Gesture ----
        gsy = bar_y + bar_h + 30
        _label(frame, "Gesture", x1+14, gsy,
               scale=0.44, color=config.PASTEL_PINK_MID)
        glabel = gesture_label if gesture_label else "—"
        _label(frame, glabel, x1+14, gsy+22,
               scale=0.52, color=config.PASTEL_ACCENT, bold=False)

        # ---- Pending tool progress ----
        if self._pending_tool and self._pending_tool != self._active:
            pct  = self._pending_frames / config.TOOL_SWITCH_DEBOUNCE_FRAMES
            pb_y = y2 - 20
            pb_w = int((pw - 28) * min(pct, 1.0))
            cv2.rectangle(frame,(x1+14,pb_y),(x1+pw-14,pb_y+6),
                          config.PASTEL_BORDER,-1)
            if pb_w > 0:
                cv2.rectangle(frame,(x1+14,pb_y),(x1+14+pb_w,pb_y+6),
                              (100,200,200),-1)
            pending_name = self._PROFILES[self._pending_tool].label
            _label(frame, f"→ {pending_name}", x1+14, pb_y-4,
                   scale=0.38, color=(100,180,180))

    # ------------------------------------------------------------------
    # COLOR PALETTE — embedded in left panel when picker is active
    # ------------------------------------------------------------------

    def draw_palette(self, frame: np.ndarray, tip: tuple | None = None):
        """
        Show color swatches in the left panel space below the tool info card.
        Only visible when tool is COLOR_PICKER.
        """
        if self._active != ToolName.COLOR_PICKER:
            self._palette_rects = []
            return

        colors  = config.PALETTE_COLORS
        sw      = config.PALETTE_SWATCH_SIZE
        gap     = config.PALETTE_SWATCH_GAP
        ncols   = 2
        nrows   = (len(colors) + ncols - 1) // ncols

        # Position palette BELOW the main left panel card (start lower)
        px  = config.LEFT_PANEL_X1 + 8
        py  = config.LEFT_PANEL_Y1 + 260   # below tool info cards
        pw  = ncols * sw + (ncols-1)*gap + 20
        ph  = nrows * (sw + gap) + 36

        # Make sure it fits in the left panel zone
        if py + ph > config.LEFT_PANEL_Y2:
            py = config.LEFT_PANEL_Y2 - ph - 4

        _pastel_card(frame, px, py, px+pw, py+ph, alpha=0.82)
        _label(frame, "COLORS", px+10, py+18,
               scale=0.44, color=config.PASTEL_PINK_MID)

        self._palette_rects = []
        for i, color in enumerate(colors):
            row = i // ncols
            col = i %  ncols
            sx  = px + 10 + col * (sw + gap)
            sy  = py + 24 + row * (sw + gap)
            x2s, y2s = sx + sw, sy + sw
            self._palette_rects.append((sx, sy, x2s, y2s))

            cv2.rectangle(frame, (sx, sy), (x2s, y2s), color, -1)
            brightness = sum(color) / 3
            border = (180,180,180) if brightness < 200 else (120,120,120)
            cv2.rectangle(frame, (sx, sy), (x2s, y2s), border, 1)

            # Active color ring
            if color == self.current_color:
                cv2.rectangle(frame,(sx-3,sy-3),(x2s+3,y2s+3),(100,200,180),2)

            # Hover progress arc
            if self._hover_swatch == i and self._hover_frames > 0:
                prog  = self._hover_frames / config.COLOR_HOVER_FRAMES
                cx_s  = (sx + x2s) // 2
                cy_s  = (sy + y2s) // 2
                r_arc = sw // 2 + 5
                angle = int(360 * prog)
                ovl   = frame.copy()
                cv2.ellipse(ovl,(cx_s,cy_s),(r_arc,r_arc),-90,0,angle,
                            (100,200,180),3)
                cv2.addWeighted(ovl,0.85,frame,0.15,0,frame)

    # ------------------------------------------------------------------
    # RIGHT PANEL — shortcuts
    # ------------------------------------------------------------------

    def draw_right_panel(self, frame: np.ndarray, fps: float = 0.0,
                         is_paused: bool = False):
        x1 = config.RIGHT_PANEL_X1 + 4
        x2 = config.RIGHT_PANEL_X2 - 4
        y1 = config.RIGHT_PANEL_Y1 + 4
        y2 = config.RIGHT_PANEL_Y2 - 4
        pw = x2 - x1

        _pastel_card(frame, x1, y1, x2, y2, alpha=0.78)

        # Title
        _label(frame, "SHORTCUTS", x1+14, y1+28,
               scale=0.48, color=config.PASTEL_PINK_MID)
        _divider(frame, x1+10, x2-10, y1+38)

        shortcuts = [
            ("Q",  "Quit"),
            ("C",  "Clear Canvas"),
            ("1",  "Pencil"),
            ("2",  "Brush"),
            ("3",  "Ink Pen"),
            ("4",  "Eraser"),
            ("5",  "Color Picker"),
        ]
        line_h = 30
        for i, (key, desc) in enumerate(shortcuts):
            ly = y1 + 58 + i * line_h
            # Key badge
            kx, ky = x1+14, ly
            cv2.rectangle(frame,(kx,ky-14),(kx+22,ky+4),
                          config.PASTEL_PINK_LIGHT,-1)
            cv2.rectangle(frame,(kx,ky-14),(kx+22,ky+4),
                          config.PASTEL_BORDER,1)
            _label(frame, key, kx+5, ky,
                   scale=0.44, color=config.PASTEL_ACCENT, bold=True)
            _label(frame, desc, kx+30, ky,
                   scale=0.44, color=config.PASTEL_PINK_DARK)

        _divider(frame, x1+10, x2-10, y1+58 + len(shortcuts)*line_h + 4)

        # Gesture map
        gy = y1 + 58 + len(shortcuts)*line_h + 22
        _label(frame, "GESTURES", x1+14, gy,
               scale=0.44, color=config.PASTEL_PINK_MID)
        gestures = [
            ("1", "Index = Pencil"),
            ("2", "2 fingers = Brush"),
            ("3", "Palm = Eraser"),
            ("4", "3 fingers = Colors"),
            ("5", "Fist = Pause"),
        ]
        for i, (icon, desc) in enumerate(gestures):
            ly2 = gy + 20 + i * 26
            _label(frame, icon, x1+14, ly2, scale=0.44,
                   color=config.PASTEL_ACCENT)
            _label(frame, desc, x1+36, ly2, scale=0.38,
                   color=config.PASTEL_PINK_DARK)

        # FPS badge at bottom of right panel
        if config.SHOW_FPS and fps > 0:
            fps_y = y2 - 14
            _label(frame, f"FPS  {fps:.0f}", x1+14, fps_y,
                   scale=0.44, color=config.PASTEL_PINK_MID)

    # ------------------------------------------------------------------
    # Compat shim — old calls from main.py still work
    # ------------------------------------------------------------------

    def draw_hud(self, frame, is_paused=False):
        self.draw_left_panel(frame, is_paused)

    def draw_shortcut_panel(self, frame):
        self.draw_right_panel(frame)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _commit_tool(self, tool: ToolName):
        if self._active != ToolName.COLOR_PICKER:
            self._prev_tool = self._active
        print(f"[Tool] → {tool.value}")
        self._active         = tool
        self._pending_tool   = None
        self._pending_frames = 0