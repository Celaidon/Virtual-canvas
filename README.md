<div align="center">

# 🐱 Vani's Canvas
### AI Gesture-Controlled Virtual Drawing Canvas

**Draw in the air. No mouse. No touch. Just your hand.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-pink?style=flat-square)

</div>

---

## ✨ What is this?

Vani's Canvas is a real-time, gesture-controlled digital drawing application. Using just your **webcam** and **hand gestures**, you can draw, erase, switch tools, and pick colors — all without touching your keyboard or mouse.

Built with Python, OpenCV, and MediaPipe, it tracks your hand 30+ times per second and translates finger poses into drawing actions on a persistent canvas overlaid on your live webcam feed.

The UI wraps everything in a custom **kawaii / pastel aesthetic** with a decorative cat-and-clouds border frame.

---

## 🖐 Gesture Controls

| Gesture | Action |
|---|---|
| ☝ **Index finger only** | Pencil — thin drawing |
| ✌ **Index + Middle fingers** | Brush — thick soft stroke |
| 🤚 **Open Palm** | Eraser — size scales with hand spread |
| 🤟 **Three fingers** | Color Picker mode |
| ✊ **Fist** | Pause drawing (lift pen) |

### Color Picker
When **three fingers** are raised, a color palette appears inside the webcam region. Hover your fingertip over any color swatch and **hold for ~0.6 seconds** to select it. The tool automatically returns to your previous drawing tool after picking.

---

## 🛠 Tools

| Tool | Keyboard | Description |
|---|---|---|
| Pencil | `1` | Thin, precise lines |
| Brush | `2` | Thick, soft strokes with slight transparency |
| Ink Pen | `3` | Smooth dark lines, always black |
| Eraser | `4` | Dynamic size based on palm spread |
| Color Picker | `5` | Hover-to-select color palette |

### Other Shortcuts

| Key | Action |
|---|---|
| `C` | Clear the canvas |
| `Q` | Quit the application |

---

## 📁 Project Structure

```
vani-canvas/
│
├── main.py               # Entry point — compositing pipeline & main loop
├── hand_tracker.py       # MediaPipe hand detection wrapper & HandData model
├── gesture_detector.py   # Gesture classification & debouncing (GestureDetector)
├── canvas_engine.py      # Persistent drawing canvas with stroke interpolation
├── toolbar.py            # ToolManager, all HUD panel rendering, color picker
├── config.py             # All constants — camera, layout, theme, thresholds
├── utils.py              # Shared helpers (text-with-bg, lerp, clamp, distance)
│
├── assets/
│   └── images/
│       └── frame.png     # Decorative border background (1920×1080)
│
├── saved_artworks/       # Auto-created; exported PNGs saved here
└── requirements.txt
```

---

## ⚙️ How It Works

```
Webcam Frame
    │
    ▼
HandTracker (MediaPipe)
    │  21 landmarks per hand → HandData
    │  (fingertip positions, fingers_up[], pinch ratios, hand_size)
    ▼
GestureDetector
    │  Raw classification → debounce window (N frames) → GestureResult
    │  (DRAW / BRUSH / OPEN_PALM / THREE_FINGER / FIST / PINCH / OK_SIGN)
    ▼
ToolManager
    │  Gesture → tool switch (with separate switch debounce)
    │  Tracks: active tool, current color, eraser radius
    ▼
CanvasEngine
    │  Draws stroke segments with sub-step interpolation (no gaps at speed)
    │  Composites strokes over webcam feed (white = transparent)
    ▼
Output Frame (1920×1080)
    │  Decorative frame background
    │  + Scaled webcam+canvas in center white zone
    │  + Left panel (tool info, color, gesture)
    │  + Right panel (shortcuts, gesture map, FPS)
    └─▶ cv2.imshow()
```

### Drawing Stability
Strokes are rendered with **sub-step circle stamping** — when the hand moves quickly between frames, the engine computes how many brush-radius steps fit along the movement vector and places overlapping circles to fill the gap. This means continuous lines even during fast gestures.

### Debouncing — two layers
- **Gesture debounce**: the raw finger pose must stay stable for N consecutive frames before the gesture is "committed." Prevents accidental triggers during hand transitions.
- **Tool-switch debounce**: a committed gesture must persist for another M frames before the active tool actually changes. Stops tools from flickering during short poses.

---

## 🚀 Installation

### Requirements
- Python 3.10 or higher
- A webcam
- Windows / macOS / Linux

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
PyQt6>=6.5.0
```

### Run

```bash
python main.py
```

### Frame image (optional but recommended)

Place your decorative border image at:
```
assets/images/frame.png
```

The image should be **1920×1080 pixels**. If the file is missing, the app runs with a plain pastel background. You can change the path in `config.py`:

```python
FRAME_IMAGE_PATH = "assets/images/frame.png"
```

---

## 🎨 Customisation

Everything tuneable lives in `config.py`. Key values:

```python
# Camera
CAMERA_INDEX  = 0        # change if your webcam isn't the default

# Drawing tools
PENCIL_THICKNESS = 2
BRUSH_THICKNESS  = 14
ERASER_THICKNESS_MIN = 30
ERASER_THICKNESS_MAX = 120

# Gesture sensitivity
GESTURE_DEBOUNCE_FRAMES     = 8   # lower = snappier, higher = more stable
TOOL_SWITCH_DEBOUNCE_FRAMES = 6

# Color picker
COLOR_HOVER_FRAMES       = 18    # ~0.6s at 30fps; increase to require longer hover
COLOR_PICKER_AUTO_RETURN = True  # return to previous tool after picking

# Palette colors (BGR)
PALETTE_COLORS = [...]            # add / remove / reorder freely
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| Camera doesn't open | Change `CAMERA_INDEX = 1` (or 2) in `config.py` |
| `AttributeError: 'HandData' has no attribute 'pinch_ratio_thumb_middle'` | Your `hand_tracker.py` is an old version — re-download it from the repo |
| Gestures feel jittery | Increase `GESTURE_DEBOUNCE_FRAMES` in `config.py` |
| Strokes have gaps | Lower `MIN_MOVE_THRESHOLD` in `config.py` |
| FPS is low | Lower `CAMERA_WIDTH / CAMERA_HEIGHT`, or set `SHOW_LANDMARKS = False` |
| Frame image not showing | Confirm `frame.png` is at `assets/images/frame.png` relative to `main.py` |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| [OpenCV](https://opencv.org/) | Webcam capture, image drawing, window display |
| [MediaPipe](https://developers.google.com/mediapipe) | Real-time hand landmark detection |
| [NumPy](https://numpy.org/) | Canvas array operations, compositing |
| [PyQt6](https://pypi.org/project/PyQt6/) | (Reserved for future native UI dialogs) |

---

## 🗺 Roadmap

- [x] Real-time hand tracking
- [x] Gesture-based tool switching
- [x] Persistent drawing canvas
- [x] Dynamic eraser (palm-size driven)
- [x] Hover-to-select color picker
- [x] Decorative frame layout
- [x] Stroke interpolation (no gaps)
- [ ] Save artwork as PNG (`S` key)
- [ ] Undo / redo (`Z` / `Y`)
- [ ] Two-hand pinch zoom
- [ ] Text tool
- [ ] Layer system

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Made with 💕 by **Vani**

*Draw freely. Draw weirdly. Draw in the air.*

</div>
