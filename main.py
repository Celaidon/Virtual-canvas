# =============================================================================
# main.py — AI Gesture Drawing Canvas  (Frame Redesign)
#
# Architecture change:
#   • Output frame is now 1920×1080 (OUTPUT_W × OUTPUT_H) — the full
#     decorative border image resolution.
#   • The webcam feed is scaled and placed inside the white center region
#     (CAM_DISPLAY_X1..X2, CAM_DISPLAY_Y1..Y2).
#   • The drawing CANVAS is kept at full 1280×720 webcam resolution internally.
#     Fingertip coordinates from MediaPipe (in webcam pixels) map directly
#     to canvas pixels — no coordinate remapping needed for drawing.
#   • For UI hit-testing (color picker swatches in left panel), fingertip
#     coords are converted FROM webcam space → output-frame space.
#   • All UI panels are drawn on the output frame OUTSIDE the webcam region,
#     inside the decorative border zones.
#   • The frame background image is composited as the base layer every frame.
# =============================================================================

import cv2
import time
import numpy as np
import os

import config
from hand_tracker     import HandTracker
from canvas_engine    import CanvasEngine
from gesture_detector import GestureDetector, GestureName
from toolbar          import ToolManager, ToolName

DRAWING_GESTURES = {GestureName.DRAW, GestureName.BRUSH, GestureName.OPEN_PALM}


# ---------------------------------------------------------------------------
# Frame background loader
# ---------------------------------------------------------------------------

def _load_frame_bg() -> np.ndarray | None:
    """
    Load the decorative border image and scale it to OUTPUT_W × OUTPUT_H.
    Returns None (with a warning) if the file is missing.
    """
    path = config.FRAME_IMAGE_PATH
    if not os.path.isfile(path):
        print(f"[WARN] Frame image not found: {path}")
        print("       Running without decorative border.")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Could not read frame image: {path}")
        return None

    if img.shape[:2] != (config.OUTPUT_H, config.OUTPUT_W):
        img = cv2.resize(img, (config.OUTPUT_W, config.OUTPUT_H),
                         interpolation=cv2.INTER_LANCZOS4)
    print(f"[INFO] Frame background loaded: {path}  {img.shape[1]}×{img.shape[0]}")
    return img


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cam_to_output(cam_pt: tuple[int, int]) -> tuple[int, int]:
    """
    Convert a webcam-space point (0..1280, 0..720) to output-frame space
    (CAM_DISPLAY region in the 1920×1080 output).
    Used for UI hit-testing (palette swatches).
    """
    cx = config.CAM_DISPLAY_X1 + int(cam_pt[0] * config.CAM_DISPLAY_W / config.CAMERA_WIDTH)
    cy = config.CAM_DISPLAY_Y1 + int(cam_pt[1] * config.CAM_DISPLAY_H / config.CAMERA_HEIGHT)
    return (cx, cy)


def cam_to_output_safe(cam_pt: tuple[int, int] | None) -> tuple[int, int] | None:
    return cam_to_output(cam_pt) if cam_pt is not None else None


# ---------------------------------------------------------------------------
# Webcam placement helpers
# ---------------------------------------------------------------------------

def _place_webcam(output: np.ndarray, cam_frame: np.ndarray):
    """
    Scale the webcam frame (1280×720) to fit CAM_DISPLAY region and
    blit it into the output frame.
    The region is CAM_DISPLAY_X1..X2, CAM_DISPLAY_Y1..Y2.
    """
    dw = config.CAM_DISPLAY_W
    dh = config.CAM_DISPLAY_H
    scaled = cv2.resize(cam_frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
    x1, y1 = config.CAM_DISPLAY_X1, config.CAM_DISPLAY_Y1
    output[y1:y1+dh, x1:x1+dw] = scaled


def _cam_cursor_to_output(cam_pt: tuple | None) -> tuple | None:
    """Map webcam-space fingertip → output-frame space for cursor drawing."""
    return cam_to_output(cam_pt) if cam_pt is not None else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Camera ----
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {config.CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {CAM_W}×{CAM_H}")

    # ---- Subsystems ----
    tracker      = HandTracker()
    canvas       = CanvasEngine(CAM_W, CAM_H)   # internal canvas = cam resolution
    detector     = GestureDetector()
    tool_manager = ToolManager()

    # ---- Decorative frame background ----
    frame_bg = _load_frame_bg()

    # ---- Window — fixed output resolution ----
    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_TITLE, config.OUTPUT_W, config.OUTPUT_H)
    if config.START_FULLSCREEN:
        cv2.setWindowProperty(config.WINDOW_TITLE,
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ---- FPS ----
    prev_time   = time.time()
    fps_display = 0.0

    print("[INFO] Ready — frame layout active.")

    # =========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ---- Detection (all in webcam/canvas space) ----
        hands_data = tracker.process_frame(frame)
        hand       = hands_data[0] if hands_data else None
        result     = detector.update(hand)

        # ---- Tool switching ----
        if result.is_active:
            tool_manager.update_from_gesture(result.name)

        active_tool = tool_manager.get_active_tool()

        if active_tool == ToolName.ERASER:
            tool_manager.update_eraser_size(hand)

        # ---- Tip in webcam/canvas space ----
        tip_cam    = hand.index_tip_smooth if hand else None
        paused     = result.is_active and result.name == GestureName.FIST

        # ---- Tip in output-frame space (for UI hit-testing) ----
        tip_output = cam_to_output_safe(tip_cam)

        # ---- Color picker — test against palette rects (output space) ----
        color_just_picked = False
        if active_tool == ToolName.COLOR_PICKER:
            color_just_picked = tool_manager.update_color_picker(tip_output)

        # ---- Drawing active? ----
        drawing_active = (
            result.is_active
            and result.name in DRAWING_GESTURES
            and not paused
            and active_tool != ToolName.COLOR_PICKER
        )

        # ---- Canvas update (webcam/canvas space — no coordinate change) ----
        canvas.update(tip_cam, drawing_active, tool_manager)

        # ---- Build composite webcam layer (cam resolution) ----
        cam_composite = canvas.composite(frame)       # strokes over webcam
        tracker.draw_landmarks(cam_composite, hands_data)
        tool_manager.draw_cursor(cam_composite, tip_cam,
                                 drawing_active, is_paused=paused)

        # ---- Build 1920×1080 output frame ----
        if frame_bg is not None:
            output = frame_bg.copy()
        else:
            # Fallback: soft pink background
            output = np.full((config.OUTPUT_H, config.OUTPUT_W, 3),
                             (235, 220, 245), dtype=np.uint8)

        # Place webcam composite into the center region
        _place_webcam(output, cam_composite)

        # ---- Draw a thin soft border around the cam region ----
        cv2.rectangle(output,
                      (config.CAM_DISPLAY_X1 - 1, config.CAM_DISPLAY_Y1 - 1),
                      (config.CAM_DISPLAY_X2 + 1, config.CAM_DISPLAY_Y2 + 1),
                      config.PASTEL_BORDER, 1)

        # ---- UI panels (drawn on output frame, outside cam region) ----
        # Gesture label for left panel
        if paused:
            gesture_label = "PAUSED"
        elif result.is_active:
            gesture_label = str(result.name)
        elif result.raw_name and str(result.raw_name) != "NONE":
            gesture_label = f"({result.raw_name})"
        else:
            gesture_label = "—"

        # FPS tracking
        now         = time.time()
        fps_display = 0.9 * fps_display + 0.1 / max(now - prev_time, 1e-9)
        prev_time   = now

        tool_manager.draw_left_panel(output, is_paused=paused,
                                     gesture_label=gesture_label)
        tool_manager.draw_palette(output, tip=tip_output)
        tool_manager.draw_right_panel(output, fps=fps_display,
                                      is_paused=paused)

        # Optional gesture debug
        if config.SHOW_GESTURE_HUD:
            detector.draw_debug(output, result,
                                pos=(config.RIGHT_PANEL_X1 + 10,
                                     config.RIGHT_PANEL_Y2 - 110))

        # ---- Show ----
        cv2.imshow(config.WINDOW_TITLE, output)

        # ---- Keys ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('c'), ord('C')):
            canvas.clear()
        elif key == ord('1'):
            tool_manager.set_active_tool(ToolName.PENCIL,       immediate=True)
        elif key == ord('2'):
            tool_manager.set_active_tool(ToolName.BRUSH,        immediate=True)
        elif key == ord('3'):
            tool_manager.set_active_tool(ToolName.INK_PEN,      immediate=True)
        elif key == ord('4'):
            tool_manager.set_active_tool(ToolName.ERASER,       immediate=True)
        elif key == ord('5'):
            tool_manager.set_active_tool(ToolName.COLOR_PICKER, immediate=True)

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()