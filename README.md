# 🌿 Silent Signals

**Plant bioelectric perception exhibits for the [Phänomena](https://phaenomena.ch) science exhibition**

Plants generate subtle electromagnetic signals that change in response to their environment. These four interactive exhibits let museum visitors discover how a living plant reacts to touch, sound, light, and human emotions — in real time.

Each exhibit uses an ESP32 microcontroller with an AD8232 biosensor to capture the plant's electrical field, then applies machine learning to classify what the plant is "perceiving."

## The Four Exhibits

| Exhibit | File | Port | What visitors do |
|---------|------|------|-----------------|
| 🤚 **Berührungswahrnehmung** (Touch) | `touch_unified_phaenomena.py` | 5004 | Touch the plant and watch it react |
| 🔔 **Klangwahrnehmung** (Sound) | `sound_unified_phaenomena.py` | 5001 | Strike a singing bowl near the plant |
| 💡 **Lichtwahrnehmung** (Light) | `lamp_unified_phaenomena.py` | 5002 | Toggle a lamp on/off to see if the plant notices |
| 😊 **Emotionswahrnehmung** (Emotion) | `emotion_unified_phaenomena.py` | 5003 | Stand in front of the camera — the plant responds to your mood |

## How It Works

```
┌─────────────┐    WebSocket    ┌─────────────┐    Flask/SocketIO    ┌─────────────┐
│   ESP32 +   │ ──────────────► │   Python     │ ──────────────────► │  Browser UI  │
│   AD8232    │   raw voltages  │   Backend    │   predictions +     │  (portrait   │
│   Sensor    │                 │   + ML       │   waveform          │   55" screen) │
└─────────────┘                 └─────────────┘                      └─────────────┘
                                      │
                                      ▼
                                 RandomForest
                                 classifier
                                 (self-training)
```

Each app follows the same architecture:
1. **ESP32 sensor** streams plant voltage readings over WebSocket at 100 Hz
2. **Python backend** collects samples, extracts features (FFT bands, statistical measures, temporal patterns), and trains a RandomForest classifier
3. **Web frontend** displays the plant's response via animated video avatars, confidence scales, and real-time signal visualization

### Self-Training Pipeline

Each exhibit has two modes:
- **Training mode**: The app automatically cycles through stimulus/rest periods, collecting labeled samples. A model is trained after ~30 paired samples (typically 10–15 minutes).
- **Exhibition mode**: The trained model classifies incoming signals in real time. Touch models persist across restarts; others retrain if the model is older than 30 minutes.

### Feature Extraction

From each 5-second voltage window, the system extracts ~50 features including:
- Statistical: mean, std, range, IQR, coefficient of variation
- Temporal: trend, segment analysis, rate of change, jitter
- Frequency: FFT power in 5 bio-relevant bands (0.05–5 Hz)
- Morphological: peak/valley counts, direction changes

## Hardware

- **Microcontroller**: ESP32 (running WebSocket server on port 81)
- **Biosensor**: AD8232 heart rate monitor (repurposed for plant bioelectric signals)
- **Plant**: Purple Heart (*Tradescantia pallida*) — chosen for strong bioelectric responses
- **Display**: 55" portrait-mounted monitor per exhibit
- **Additional**: Shelly smart plug (lamp exhibit), USB solenoid (sound exhibit), webcam (emotion + gesture detection)

## Installation

```bash
pip install flask flask-socketio websockets scikit-learn scipy numpy

# Run any exhibit
python phaenomena/touch_unified_phaenomena.py
python phaenomena/sound_unified_phaenomena.py
python phaenomena/lamp_unified_phaenomena.py
python phaenomena/emotion_unified_phaenomena.py
```

### Configuration

Each app has ESP32 connection settings at the top of the file:
```python
ESP32_IP = "192.168.1.133"  # Adjust to your sensor's IP
ESP32_PORT = 81
```

### Fonts

The UI uses Phänomena corporate design fonts. Place them in `./fonts/Fonts/`:
- `TownandCountryJNL/` — titles
- `Satoshi Fonts/` — body text

### Plant Avatar Videos

Place looping `.mp4` avatar videos in `./plant-viz-movies/`:
- Touch: `hand.mp4`, `nohand.mp4`
- Sound: `sound.mp4`, `nosound.mp4`
- Light: `light.mp4`, `nolight.mp4`
- Emotion: `positive.mp4`, `negative.mp4`, `neutral.mp4`

## UI Features

- **4-language support**: DE / EN / FR / IT (gesture swipe or button tap)
- **Gesture control**: Right-to-left swipe via MediaPipe pose detection cycles language
- **Staff controls**: Hidden bottom-right buttons for Training / Exhibition / Reset
- **QR panel**: Shows ESP32 WebSocket address for the Plantangle companion app

## Research Context

This project is part of ongoing research at the [MIT Center for Collective Intelligence](https://cci.mit.edu) and [galaxyadvisors AG](https://galaxyadvisors.com) into plant-human bioelectric interactions, exploring whether plants exhibit measurable responses to environmental stimuli that can be decoded through machine learning.

## Exhibition

**Living Synthesis** — opening March 2026 at [Phänomena](https://phaenomena.ch), Zürich

## License

MIT
