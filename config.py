# =============================================================================
# config.py — Central configuration for AI Gesture Drawing Canvas
# =============================================================================

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
TARGET_FPS    = 30

# ---------------------------------------------------------------------------
# Hand tracking (MediaPipe)
# ---------------------------------------------------------------------------
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.75
TRACKING_CONFIDENCE  = 0.65

# ---------------------------------------------------------------------------
# Smoothing / jitter reduction
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW   = 7
MIN_MOVE_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
WINDOW_TITLE     = "Vani's Canvas"
START_FULLSCREEN = False
CANVAS_BG_COLOR  = (255, 255, 255)

# ---------------------------------------------------------------------------
# Decorative frame layout
# ---------------------------------------------------------------------------
# The frame image is 1920x1080.  All output is rendered at this resolution.
FRAME_IMAGE_PATH = "assets/images/frame.png"
OUTPUT_W = 1920
OUTPUT_H = 1080

# Webcam/canvas display region — 16:9, centered in the white zone
# White zone: roughly x=200..1720, y=256..700
# Webcam: 782x440 centered at x=960, y=260..700
CAM_DISPLAY_W  = 1280
CAM_DISPLAY_H  = 720
CAM_DISPLAY_X1 = 320   # left edge of webcam rect in output frame
CAM_DISPLAY_Y1 = 170    # top edge
CAM_DISPLAY_X2 = CAM_DISPLAY_X1+CAM_DISPLAY_W   # right edge1600
CAM_DISPLAY_Y2 = CAM_DISPLAY_Y1+CAM_DISPLAY_H    # bottom edge890

# Scale factors: webcam pixel → canvas pixel
# (used to map fingertip from webcam coords → canvas coords)
CAM_SCALE_X = CAMERA_WIDTH  / CAM_DISPLAY_W   # 1280 / 782 ≈ 1.637
CAM_SCALE_Y = CAMERA_HEIGHT / CAM_DISPLAY_H   # 720  / 440 ≈ 1.636

# Left UI panel region (x: 8 → CAM_DISPLAY_X1-8)
LEFT_PANEL_X1 = 8
LEFT_PANEL_X2 = CAM_DISPLAY_X1 - 8    # 561
LEFT_PANEL_Y1 = CAM_DISPLAY_Y1        # 260
LEFT_PANEL_Y2 = CAM_DISPLAY_Y2        # 700
LEFT_PANEL_W  = LEFT_PANEL_X2 - LEFT_PANEL_X1   # 553
LEFT_PANEL_H  = LEFT_PANEL_Y2 - LEFT_PANEL_Y1   # 440

# Right UI panel region (x: CAM_DISPLAY_X2+8 → 1912)
RIGHT_PANEL_X1 = CAM_DISPLAY_X2 + 8   # 1359
RIGHT_PANEL_X2 = 1912
RIGHT_PANEL_Y1 = CAM_DISPLAY_Y1       # 260
RIGHT_PANEL_Y2 = CAM_DISPLAY_Y2       # 700
RIGHT_PANEL_W  = RIGHT_PANEL_X2 - RIGHT_PANEL_X1  # 553
RIGHT_PANEL_H  = RIGHT_PANEL_Y2 - RIGHT_PANEL_Y1  # 440

# ---------------------------------------------------------------------------
# Pastel / kawaii theme colors (BGR)
# ---------------------------------------------------------------------------
PASTEL_PINK_DARK   = (180, 160, 210)   # dusty rose text
PASTEL_PINK_MID    = (200, 190, 230)   # soft labels
PASTEL_PINK_LIGHT  = (235, 220, 245)   # card backgrounds
PASTEL_CARD_BG     = (245, 235, 250)   # panel card fill (very light lavender)
PASTEL_ACCENT      = (160, 120, 210)   # accent text / highlights
PASTEL_WHITE       = (255, 255, 255)
PASTEL_BORDER      = (210, 185, 225)   # card border / dividers

TOOL_COLORS = {
    "pencil"      : (140, 200, 140),   # soft green
    "brush"       : (140, 180, 255),   # soft orange-blue
    "ink_pen"     : (160, 160, 160),   # grey
    "eraser"      : (180, 140, 210),   # soft purple
    "color_picker": (200, 220, 100),   # soft teal-lime
}

# ---------------------------------------------------------------------------
# Tool drawing parameters
# ---------------------------------------------------------------------------
PENCIL_THICKNESS = 2

BRUSH_THICKNESS  = 14
BRUSH_OPACITY    = 0.90

INKPEN_THICKNESS = 4
INKPEN_COLOR     = (0, 0, 0)
INKPEN_OPACITY   = 1.0

ERASER_THICKNESS_MIN = 30
ERASER_THICKNESS_MAX = 120
ERASER_HAND_SCALE    = 3.5

DEFAULT_TOOL  = "pencil"
DEFAULT_COLOR = (0, 0, 0)

# ---------------------------------------------------------------------------
# Gesture thresholds
# ---------------------------------------------------------------------------
GESTURE_DEBOUNCE_FRAMES = 8
PINCH_CLOSE_RATIO = 0.07
PINCH_OPEN_RATIO  = 0.13
OK_PINCH_RATIO    = 0.09
TOOL_SWITCH_DEBOUNCE_FRAMES = 6

# ---------------------------------------------------------------------------
# Color picker
# ---------------------------------------------------------------------------
COLOR_HOVER_FRAMES       = 18
COLOR_PICKER_AUTO_RETURN = True

# BGR palette — shown in left panel when color picker is active
PALETTE_COLORS = [
    (0,   0,   200),   # Red
    (200, 80,  80 ),   # Blue
    (60,  180, 60 ),   # Green
    (0,   0,   0  ),   # Black
    (255, 255, 255),   # White
    (180, 140, 210),   # Pink
    (180, 60,  120),   # Purple
    (0,   210, 230),   # Yellow
    (30,  100, 160),   # Brown
]
PALETTE_SWATCH_SIZE = 46
PALETTE_SWATCH_GAP  = 8

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
SAVE_FOLDER = "saved_artworks"

# ---------------------------------------------------------------------------
# Debug / display flags
# ---------------------------------------------------------------------------
SHOW_FPS         = True
SHOW_LANDMARKS   = True
SHOW_GESTURE_HUD = False
SHOW_TOOL_HUD    = True