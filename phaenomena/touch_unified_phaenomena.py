#!/usr/bin/env python3
"""
🌿 Phaenomena - BERÜHRUNGSWAHRNEHMUNG (Touch Perception)
Unified Exhibition Interface with Gesture Controls


Features:
- Phänomena Corporate Design (colors, Satoshi font)
- Gesture control for language selection (swipe left/right)  
- Single-person tracking with foot position guide
- First-person plant voice
- 4-language support (DE/FR/IT/EN)


Port: 5004
Display: Optimized for 55" portrait monitor
"""


import asyncio
import json
import pickle
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
import threading
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


from flask import Flask, render_template_string, jsonify, send_from_directory
from flask_socketio import SocketIO
import websockets


# ===== CONFIGURATION =====
ESP32_IP = "192.168.1.133"
ESP32_PORT = 81


SAMPLE_RATE = 100
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS


WARMUP_PAIRS = 20
MAX_TRAINING_PAIRS = 100
MIN_PAIRS_FOR_RETRAIN = 15


PING_MIN_INTERVAL = 10
PING_MAX_INTERVAL = 20


POST_PING_CAPTURE_START = 0
POST_PING_CAPTURE_END = 5
NOTOUCH_CAPTURE_START = 7
NOTOUCH_CAPTURE_END = 12


UPDATE_RATE_MS = 200
PREDICTION_INTERVAL = 0.3


OUTPUT_DIR = Path("./touch_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)


VIDEO_DIR = Path("./plant-viz-movies")
FONTS_DIR = Path("./fonts/Fonts")
FONTS_BASE = Path("./fonts")


import wave


def save_wav_file(samples_v, label, event_number):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plant_{label}_e{event_number:03d}_{timestamp}.wav"
    filepath = OUTPUT_DIR / filename
    
    samples_array = np.array(samples_v, dtype=np.float64)
    normalized = 2 * (samples_array - 0.5) / 2.5 - 1
    normalized = np.clip(normalized, -1, 1)
    int_samples = (normalized * 32767).astype(np.int16)
    
    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(int_samples.tobytes())
    return filepath




# ===== FEATURE EXTRACTION =====
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


def extract_features(samples, sr=SAMPLE_RATE):
    features = {}
    samples = np.array(samples).astype(np.float64)
    n = len(samples)
    if n < 100:
        return None
    
    eps = 1e-10
    features['mean_voltage'] = float(np.mean(samples))
    features['voltage_std'] = float(np.std(samples) + eps)
    features['voltage_range'] = float(np.ptp(samples))
    
    try:
        peaks, props = find_peaks(samples, height=np.mean(samples), distance=sr//10, prominence=0.01)
        features['n_peaks'] = float(len(peaks))
        features['max_peak_prominence'] = float(np.max(props.get('prominences', [0]))) if len(peaks) > 0 else 0
    except:
        features['n_peaks'] = 0
        features['max_peak_prominence'] = 0
    
    diff1 = np.diff(samples)
    features['max_derivative'] = float(np.max(np.abs(diff1)))
    features['mean_abs_derivative'] = float(np.mean(np.abs(diff1)))
    
    fft_vals = np.abs(fft(samples))[:n//2]
    freqs = fftfreq(n, 1/sr)[:n//2]
    total_power = np.sum(fft_vals ** 2) + eps
    
    for name, low, high in [('slow', 0.2, 0.5), ('mid', 1.0, 2.0), ('fast', 2.0, 5.0)]:
        mask = (freqs >= low) & (freqs < high)
        features[f'power_{name}'] = float(np.sum(fft_vals[mask] ** 2) / total_power) if np.any(mask) else 0
    
    return features




# ===== ML MODEL =====
class AdaptiveTouchModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        self.scaler = StandardScaler()
        self.touch_features = []
        self.notouch_features = []
        self.is_trained = False
        self.accuracy_estimate = 0.0
        self.warmup_progress = 0.0
        self.feature_names = None
        self.model_file = OUTPUT_DIR / "touch_model_latest.pkl"
    
    def add_sample(self, features, is_touch):
        lst = self.touch_features if is_touch else self.notouch_features
        lst.append(features)
        if len(lst) > MAX_TRAINING_PAIRS:
            lst.pop(0)
        
        total = len(self.touch_features) + len(self.notouch_features)
        self.warmup_progress = min(1.0, total / (WARMUP_PAIRS * 2))
        
        if len(self.touch_features) >= WARMUP_PAIRS and len(self.notouch_features) >= WARMUP_PAIRS:
            return self._train()
        return False
    
    def _train(self):
        if len(self.touch_features) < MIN_PAIRS_FOR_RETRAIN:
            return False
        
        self.feature_names = list(self.touch_features[0].keys())
        X_touch = [[f[k] for k in self.feature_names] for f in self.touch_features]
        X_notouch = [[f[k] for k in self.feature_names] for f in self.notouch_features]
        
        X = np.array(X_touch + X_notouch)
        y = np.array([1] * len(X_touch) + [0] * len(X_notouch))
        
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.accuracy_estimate = self.model.score(X_scaled, y)
        self.is_trained = True
        self._save()
        print(f"✨ Model trained: {self.accuracy_estimate:.1%} accuracy")
        return True
    
    def predict(self, features):
        if not self.is_trained:
            return None, 0.0
        try:
            X = np.array([[features[k] for k in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            conf = float(max(self.model.predict_proba(X_scaled)[0]))
            return pred, conf
        except:
            return None, 0.0
    
    def force_retrain(self):
        self.touch_features = []
        self.notouch_features = []
        self.is_trained = False
        self.warmup_progress = 0.0
    
    def _save(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 
                        'feature_names': self.feature_names, 'accuracy': self.accuracy_estimate,
                        'timestamp': datetime.now().isoformat()}, f)
    
    def load_latest(self):
        if not self.model_file.exists():
            return False, 0
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.accuracy_estimate = data['accuracy']
            self.is_trained = True
            age = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds() / 60
            return True, age
        except:
            return False, 0




# ===== STATE =====
class AppState:
    def __init__(self):
        self.connected = False
        self.voltage_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.current_voltage = 0
        self.mode = 'training'
        self.prediction = 'resting'
        self.confidence = 0.0
        self.model = AdaptiveTouchModel()
        self.last_ping_time = 0
        self.next_ping_interval = PING_MIN_INTERVAL
        self.ping_active = False
        self.ping_start_time = 0
        self.collecting_touch = False
        self.collecting_notouch = False
        self.touch_samples = []
        self.notouch_samples = []
        self.touch_count = 0
        self.notouch_count = 0
        self.event_number = 0
        self.last_prediction_time = 0


state = AppState()




_diag_counter = 0

def process_voltage_for_training(voltages, current_time):
    global _diag_counter
    _diag_counter += 1
    if _diag_counter % 50 == 1:  # Print every 50th call
        time_since_ping = current_time - state.last_ping_time
        print(f"  [DIAG] mode={state.mode} ping_active={state.ping_active} "
              f"time_since_last_ping={time_since_ping:.1f}s next_interval={state.next_ping_interval:.1f}s "
              f"n_voltages={len(voltages)}")
    
    if state.mode != 'training':
        return
    
    if not state.ping_active:
        time_since = current_time - state.last_ping_time
        if time_since >= state.next_ping_interval:
            state.ping_active = True
            state.ping_start_time = current_time
            state.next_ping_interval = random.uniform(PING_MIN_INTERVAL, PING_MAX_INTERVAL)
            state.last_ping_time = current_time
            state.event_number += 1
            print(f"  🟢 Cycle #{state.event_number}: TOUCH phase — visitor should touch plant")
    else:
        t = current_time - state.ping_start_time
        
        if POST_PING_CAPTURE_START <= t < POST_PING_CAPTURE_END:
            if not state.collecting_touch:
                state.collecting_touch = True
                state.touch_samples = []
            state.touch_samples.extend(voltages)
        elif t >= POST_PING_CAPTURE_END and state.collecting_touch:
            state.collecting_touch = False
            n_samples = len(state.touch_samples)
            print(f"  🔴 RELEASE — collected {n_samples} touch samples")
            if n_samples >= 100:
                samples_v = [v / 1000.0 for v in state.touch_samples]
                state.touch_count += 1
                save_wav_file(samples_v, "touch", state.touch_count)
                features = extract_features(np.array(samples_v))
                if features:
                    state.model.add_sample(features, True)
                    print(f"  ✅ Touch sample #{state.touch_count} added to model")
                else:
                    print(f"  ⚠️  Feature extraction failed for touch sample")
            else:
                print(f"  ⚠️  Too few samples ({n_samples} < 100)")
        
        if NOTOUCH_CAPTURE_START <= t < NOTOUCH_CAPTURE_END:
            if not state.collecting_notouch:
                state.collecting_notouch = True
                state.notouch_samples = []
            state.notouch_samples.extend(voltages)
        elif t >= NOTOUCH_CAPTURE_END and state.collecting_notouch:
            state.collecting_notouch = False
            state.ping_active = False
            n_samples = len(state.notouch_samples)
            print(f"  📊 Collected {n_samples} no-touch samples")
            if n_samples >= 100:
                samples_v = [v / 1000.0 for v in state.notouch_samples]
                state.notouch_count += 1
                save_wav_file(samples_v, "notouch", state.notouch_count)
                features = extract_features(np.array(samples_v))
                if features:
                    state.model.add_sample(features, False)
                    print(f"  ✅ No-touch sample #{state.notouch_count} added | Next cycle in {state.next_ping_interval:.0f}s")
                else:
                    print(f"  ⚠️  Feature extraction failed for no-touch sample")
            else:
                print(f"  ⚠️  Too few samples ({n_samples} < 100)")




# ===== FLASK APP =====
app = Flask(__name__)
app.config['SECRET_KEY'] = 'phaenomena-touch-2025'


import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)


socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')




HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>BERÜHRUNGSWAHRNEHMUNG - Phänomena</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <style>
        @font-face { font-family: 'TownAndCountryJNL'; src: url('/fonts/TownandCountryJNL.otf') format('opentype'), url('/fonts/TownAndCountryJNL.otf') format('opentype'), url('/fonts/Town and Country JNL.otf') format('opentype'), url('/fonts/TownandCountryJNL-Regular.otf') format('opentype'), url('/fonts/Town%20and%20Country%20JNL.otf') format('opentype'); font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Light.otf') format('opentype'), url('/fonts/Satoshi-Light.ttf') format('truetype'); font-weight: 300; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Regular.otf') format('opentype'), url('/fonts/Satoshi-Regular.ttf') format('truetype'); font-weight: 400; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Medium.otf') format('opentype'), url('/fonts/Satoshi-Medium.ttf') format('truetype'); font-weight: 500; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Bold.otf') format('opentype'), url('/fonts/Satoshi-Bold.ttf') format('truetype'); font-weight: 700; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Black.otf') format('opentype'), url('/fonts/Satoshi-Black.ttf') format('truetype'); font-weight: 900; font-display: swap; }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --dark: #201844; --green: #1cffa8; --cyan: #2fd6ee; --pink: #ff6ac8; }
        
        body { background: var(--dark); color: #fff; font-family: 'Satoshi', sans-serif; min-height: 100vh; overflow: hidden; }
        
        .container { display: flex; flex-direction: column; align-items: center; justify-content: flex-start; height: 100vh; padding: 4vh 30px 0; gap: 0; }
        
        .header { text-align: center; flex-shrink: 0; margin-bottom: 5vh; }
        .exhibit-title { font-family: 'TownAndCountryJNL', serif; font-size: 3.2em; font-weight: 400; letter-spacing: 0.08em; color: var(--green); margin-bottom: 16px; }
        .exhibit-subtitle { font-family: 'Satoshi', sans-serif; font-size: 1.6em; font-weight: 400; color: rgba(255,255,255,0.9); }
        
        .connection-dot { position: fixed; top: 25px; right: 25px; width: 14px; height: 14px; border-radius: 50%; background: #ef4444; z-index: 100; }
        .connection-dot.connected { background: var(--green); box-shadow: 0 0 15px var(--green); animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 15px var(--green); } 50% { box-shadow: 0 0 25px var(--green); } }
        
        .plant-panel { display: flex; flex-direction: column; align-items: center; gap: 4.5vh; flex: 1; justify-content: center; width: 100%; padding-bottom: 380px; }
        
        .video-container { width: 420px; height: 420px; border-radius: 50%; overflow: hidden; border: 4px solid var(--green); box-shadow: 0 0 60px rgba(28,255,168,0.3); transition: border-color 0.4s, box-shadow 0.4s; }
        .video-container.touch-active { border-color: var(--cyan); box-shadow: 0 0 100px rgba(47,214,238,0.5); }
        .video-container.ping-active { border-color: var(--pink); box-shadow: 0 0 100px rgba(255,106,200,0.5); animation: ping-pulse 0.6s infinite; }
        @keyframes ping-pulse { 0%,100% { box-shadow: 0 0 80px rgba(255,106,200,0.4); } 50% { box-shadow: 0 0 120px rgba(255,106,200,0.6); } }
        .video-container video { width: 100%; height: 100%; object-fit: cover; transform: scale(1.15); }
        
        .plant-response { font-family: 'Satoshi', sans-serif; font-size: 2.5em; font-weight: 600; text-align: center; min-height: 1.3em; }
        .plant-response.resting { color: var(--green); }
        .plant-response.touch { color: var(--cyan); text-shadow: 0 0 30px var(--cyan); }
        .plant-response.ping-active { color: var(--pink) !important; text-shadow: 0 0 30px var(--pink); }
        
        /* 4-segment confidence scale - wider, more centered */
        .confidence-section { display: flex; flex-direction: column; align-items: center; gap: 12px; width: 100%; max-width: 600px; margin: 0 auto; }
        .confidence-value { font-family: 'Satoshi', sans-serif; font-size: 1.15em; font-weight: 600; color: rgba(255,255,255,0.8); min-height: 1.3em; text-align: center; }
        .confidence-scale { position: relative; width: 100%; height: 16px; border-radius: 8px; }
        .confidence-scale-bg { position: absolute; inset: 0; display: flex; border-radius: 8px; overflow: hidden; }
        .confidence-scale-bg .seg { flex: 1; }
        .confidence-scale-bg .seg:nth-child(1) { background: var(--pink); }
        .confidence-scale-bg .seg:nth-child(2) { background: rgba(255,106,200,0.4); }
        .confidence-scale-bg .seg:nth-child(3) { background: rgba(28,255,168,0.4); }
        .confidence-scale-bg .seg:nth-child(4) { background: var(--green); }
        .confidence-needle { position: absolute; top: -5px; width: 5px; height: 26px; background: #fff; border-radius: 2px; transition: left 0.4s ease; box-shadow: 0 0 10px rgba(255,255,255,0.7); z-index: 2; }
        .confidence-labels { display: flex; justify-content: space-between; width: 100%; font-family: 'Satoshi', sans-serif; font-size: 0.82em; color: rgba(255,255,255,0.55); margin-top: 4px; }
        .confidence-labels span { text-align: center; flex: 1; line-height: 1.3; }
        
        /* ===== FIXED BOTTOM: signal bar (full width) ===== */
        .signal-container { position: fixed; bottom: 200px; left: 0; right: 0; height: 100px; background: rgba(255,255,255,0.03); padding: 8px 20px; border-top: 1px solid rgba(28,255,168,0.15); border-bottom: 1px solid rgba(28,255,168,0.15); z-index: 80; }
        .signal-title { font-family: 'Satoshi', sans-serif; font-size: 0.7em; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 2px; }
        #signalCanvas { display: block; width: 100%; height: 72px; }
        
        /* ===== FIXED BOTTOM BAR: stickman on top, lang below | staff + QR (right) ===== */
        .bottom-bar { position: fixed; bottom: 0; left: 0; right: 0; height: 190px; display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; z-index: 90; background: rgba(32,24,68,0.6); }
        
        .language-section { display: flex; flex-direction: column; align-items: center; gap: 8px; }
        .gesture-overlay { width: 200px; height: 110px; border-radius: 10px; overflow: hidden; border: 2px solid rgba(28,255,168,0.4); background: rgba(0,0,0,0.5); transition: all 0.3s; flex-shrink: 0; }
        .gesture-overlay:hover { border-color: var(--green); }
        .gesture-overlay.detected { border-color: var(--green); box-shadow: 0 0 15px rgba(28,255,168,0.3); }
        #gestureVideo { display: none; }
        #gestureCanvas { width: 100%; height: 100%; }
        
        .lang-group { display: flex; flex-direction: column; align-items: center; gap: 4px; }
        .swipe-hint { font-family: 'Satoshi', sans-serif; font-size: 0.65em; color: rgba(255,255,255,0.35); }
        .language-float { display: flex; gap: 8px; }
        .lang-btn { padding: 8px 18px; border-radius: 20px; font-size: 1em; font-weight: 600; background: rgba(32,24,68,0.9); border: 2px solid rgba(255,255,255,0.2); color: rgba(255,255,255,0.5); cursor: pointer; transition: all 0.3s; font-family: 'Satoshi', sans-serif; }
        .lang-btn:hover { border-color: rgba(255,255,255,0.4); color: rgba(255,255,255,0.8); }
        .lang-btn.active { background: var(--green); border-color: var(--green); color: var(--dark); }
        
        .gesture-feedback { position: fixed; bottom: 310px; left: 20px; width: 200px; text-align: center; font-size: 1.4em; font-weight: 700; color: var(--green); text-shadow: 0 0 20px var(--green); opacity: 0; pointer-events: none; z-index: 95; }
        .gesture-feedback.active { opacity: 1; animation: pop 0.5s ease-out; }
        @keyframes pop { 0% { transform: scale(0.5); opacity: 0; } 50% { transform: scale(1.1); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
        
        /* Right side of bottom bar: staff buttons + QR */
        .bottom-right { display: flex; flex-direction: column; align-items: flex-end; gap: 10px; justify-content: center; }
        
        /* Staff controls - visible enough to find */
        .staff-controls { display: flex; gap: 6px; opacity: 0.35; transition: opacity 0.3s; }
        .staff-controls:hover { opacity: 1; }
        .staff-btn { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.25); color: rgba(255,255,255,0.7); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-family: 'Satoshi', sans-serif; font-size: 0.75em; }
        .staff-btn:hover { background: rgba(255,255,255,0.2); }
        
        /* QR - compact */
        .qr-panel { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: 10px; background: rgba(0,0,0,0.3); border: 1px solid rgba(28,255,168,0.25); }
        .qr-panel h4 { display: none; }
        .qr-info { display: flex; flex-direction: column; }
        .qr-label { font-size: 0.55em; color: rgba(255,255,255,0.55); word-break: break-all; }
        .qr-hint { font-size: 0.5em; color: rgba(255,255,255,0.3); }
        
        .training-stats-bar { display: none; position: fixed; bottom: 0; left: 0; right: 0; background: rgba(64,49,138,0.95); padding: 10px 25px; justify-content: center; gap: 35px; font-size: 0.95em; z-index: 150; }
        .training-stats-bar.visible { display: flex; }
        .training-stat { color: rgba(255,255,255,0.8); }
        .training-stat strong { color: var(--green); margin-left: 5px; }
        .training-progress { width: 100px; height: 5px; background: rgba(255,255,255,0.2); border-radius: 3px; overflow: hidden; }
        .training-progress-fill { height: 100%; background: var(--green); transition: width 0.5s; }
        
        .no-camera { display: flex; align-items: center; justify-content: center; height: 100%; font-size: 0.7em; color: rgba(255,255,255,0.4); }
    </style>
</head>
<body>
    <div class="connection-dot" id="connectionDot"></div>
    
    <div class="container">
        <div class="header">
            <div class="exhibit-title" id="exhibitTitle">BERÜHRUNGSWAHRNEHMUNG</div>
            <div class="exhibit-subtitle" id="exhibitSubtitle">Spüre ich deine Berührung? Probiere es aus.</div>
        </div>
        
        <div class="plant-panel">
            <div class="video-container" id="videoContainer">
                <video id="plantVideo" autoplay loop muted playsinline>
                    <source src="/videos/nohand.mp4" type="video/mp4">
                </video>
            </div>
            <div class="plant-response resting" id="plantResponse">Ich ruhe...</div>
            
            <div class="confidence-section">
                <div class="confidence-value" id="confidenceValue">--</div>
                <div class="confidence-scale">
                    <div class="confidence-scale-bg">
                        <div class="seg"></div><div class="seg"></div><div class="seg"></div><div class="seg"></div>
                    </div>
                    <div class="confidence-needle" id="confidenceNeedle" style="left:50%"></div>
                </div>
                <div class="confidence-labels" id="confidenceLabels">
                    <span>Berührung<br>sicher</span>
                    <span>Berührung<br>wahrscheinlich</span>
                    <span>Ruhe<br>wahrscheinlich</span>
                    <span>Ruhe<br>sicher</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Full-width signal strip -->
    <div class="signal-container">
        <div class="signal-title" id="signalTitle">Elektromagnetisches Feld der Pflanze</div>
        <canvas id="signalCanvas"></canvas>
    </div>
    
    <!-- Bottom bar -->
    <div class="bottom-bar">
        <div class="language-section">
            <div class="gesture-overlay" id="gestureOverlay">
                <video id="gestureVideo" playsinline></video>
                <canvas id="gestureCanvas"></canvas>
            </div>
            <div class="lang-group">
                <div class="swipe-hint" id="swipeHint">👋 Wische für Sprache</div>
                <div class="language-float" id="languageFloat">
                    <button class="lang-btn active" data-lang="de">DE</button>
                    <button class="lang-btn" data-lang="en">EN</button>
                    <button class="lang-btn" data-lang="fr">FR</button>
                    <button class="lang-btn" data-lang="it">IT</button>
                </div>
            </div>
        </div>
        
        <div class="bottom-right">
            <div class="staff-controls">
                <button class="staff-btn" id="toggleModeBtn">Training</button>
                <button class="staff-btn" id="retrainBtn">Retrain</button>
            </div>
            <div class="qr-panel">
                <h4>Plantangle QR</h4>
                <div id="qrBox"></div>
                <div class="qr-info">
                    <div class="qr-label" id="qrLabel"></div>
                    <div class="qr-hint">Scan to auto-fill IP:PORT</div>
                </div>
            </div>
        </div>
    </div>
    <div class="gesture-feedback" id="gestureFeedback"></div>
    
    <div class="training-stats-bar" id="trainingStatsBar">
        <div class="training-stat">Berührung: <strong id="touchCount">0</strong></div>
        <div class="training-stat">Ruhe: <strong id="restCount">0</strong></div>
        <div class="training-stat">Genauigkeit: <strong id="accuracy">--</strong></div>
        <div class="training-stat" style="display:flex;align-items:center;gap:6px;">
            <span style="color:rgba(255,255,255,0.6)">Fortschritt:</span>
            <div class="training-progress"><div class="training-progress-fill" id="trainingProgressFill"></div></div>
        </div>
    </div>
    
    <script>
        const plantWs = "{{ plant_ws }}";
        const socket = io();
        let currentLang = 'de', currentMode = 'exhibition', lastPingActive = false;
        const languages = ['de', 'en', 'fr', 'it'];
        
        const T = {
            de: {
                title: 'BERÜHRUNGSWAHRNEHMUNG',
                subtitle: 'Spüre ich deine Berührung? Probiere es aus.',
                resting: 'Ich ruhe...', touch: 'Ich spüre dich!', touchNow: 'Berühre mich jetzt!', letGo: 'Lass mich los!',
                signalTitle: 'Elektromagnetisches Feld der Pflanze',
                swipeHint: '👋 Wische für Sprache',
                confLabels: ['Berührung|sicher', 'Berührung|wahrscheinlich', 'Ruhe|wahrscheinlich', 'Ruhe|sicher']
            },
            en: {
                title: 'TOUCH PERCEPTION',
                subtitle: 'Can I feel your touch? Try it out.',
                resting: 'I am resting...', touch: 'I feel you!', touchNow: 'Touch me now!', letGo: 'Let go of me!',
                signalTitle: 'Electromagnetic field of the plant',
                swipeHint: '👋 Swipe for language',
                confLabels: ['Touch|certain', 'Touch|probable', 'Rest|probable', 'Rest|certain']
            },
            fr: {
                title: 'PERCEPTION DU TOUCHER',
                subtitle: 'Est-ce que je sens ton toucher? Essaie.',
                resting: 'Je me repose...', touch: 'Je te sens!', touchNow: 'Touche-moi maintenant!', letGo: 'Lâche-moi!',
                signalTitle: 'Champ électromagnétique de la plante',
                swipeHint: '👋 Glisse pour langue',
                confLabels: ['Toucher|certain', 'Toucher|probable', 'Repos|probable', 'Repos|certain']
            },
            it: {
                title: 'PERCEZIONE DEL TATTO',
                subtitle: 'Posso sentire il tuo tocco? Provaci.',
                resting: 'Sto riposando...', touch: 'Ti sento!', touchNow: 'Toccami ora!', letGo: 'Lasciami!',
                signalTitle: 'Campo elettromagnetico della pianta',
                swipeHint: '👋 Scorri per lingua',
                confLabels: ['Tocco|certo', 'Tocco|probabile', 'Riposo|probabile', 'Riposo|certo']
            }
        };
        
        const t = k => T[currentLang]?.[k] || T.en[k] || k;
        
        function updateConfLabels() {
            const labels = t('confLabels');
            document.getElementById('confidenceLabels').innerHTML = labels.map(l => {
                const parts = l.split('|');
                return '<span>' + parts[0] + '<br>' + parts[1] + '</span>';
            }).join('');
        }
        
        function updateLang(lang) {
            currentLang = lang;
            document.querySelectorAll('.lang-btn').forEach(e => e.classList.toggle('active', e.dataset.lang === lang));
            document.getElementById('exhibitTitle').textContent = t('title');
            document.getElementById('exhibitSubtitle').textContent = t('subtitle');
            document.getElementById('signalTitle').textContent = t('signalTitle');
            document.getElementById('swipeHint').textContent = t('swipeHint');
            updateConfLabels();
        }
        
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => updateLang(btn.dataset.lang));
        });
        
        function cycleLang(dir) {
            const idx = languages.indexOf(currentLang);
            updateLang(languages[(idx + dir + 4) % 4]);
            showFeedback('🌐 ' + currentLang.toUpperCase());
        }
        
        document.addEventListener('keydown', e => {
            if (e.key >= '1' && e.key <= '4') updateLang(languages[e.key - 1]);
            if (e.key === 'ArrowLeft') cycleLang(-1);
            if (e.key === 'ArrowRight') cycleLang(1);
        });
        
        // Gesture control
        const gestureVideo = document.getElementById('gestureVideo');
        const gestureCanvas = document.getElementById('gestureCanvas');
        const gestureCtx = gestureCanvas.getContext('2d');
        const gestureOverlay = document.getElementById('gestureOverlay');
        
        let handHistory = { left: [], right: [] }, langCooldown = false, animPhase = 0;
        const swipeThreshold = 0.8, histLen = 12;
        const colors = { stem: '#2d6b2d', leaf: '#3d8b3d', flower: '#e8b4d8', center: '#f4d03f' };
        
        async function initGesture() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
                gestureVideo.srcObject = stream;
                await gestureVideo.play();
                gestureCanvas.width = gestureVideo.videoWidth;
                gestureCanvas.height = gestureVideo.videoHeight;
                
                const pose = new Pose({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}` });
                pose.setOptions({ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                pose.onResults(onPose);
                
                (async function loop() { await pose.send({ image: gestureVideo }); requestAnimationFrame(loop); })();
            } catch (e) {
                gestureOverlay.innerHTML = '<div class="no-camera">📷</div>';
            }
        }
        
        function onPose(results) {
            const w = gestureCanvas.width, h = gestureCanvas.height;
            animPhase += 0.03;
            gestureCtx.save();
            gestureCtx.clearRect(0, 0, w, h);
            gestureCtx.translate(w, 0);
            gestureCtx.scale(-1, 1);
            gestureCtx.fillStyle = '#0a0f14';
            gestureCtx.fillRect(0, 0, w, h);
            
            let detected = false;
            if (results.poseLandmarks) {
                const lm = results.poseLandmarks;
                const cx = (lm[11].x + lm[12].x) / 2;
                const vis = (lm[11].visibility + lm[12].visibility) / 2;
                if (vis > 0.5 && Math.abs(cx - 0.5) < 0.4) {
                    detected = true;
                    drawAvatar(lm, w, h);
                    detectGestures(lm);
                }
            }
            gestureOverlay.classList.toggle('detected', detected);
            gestureCtx.restore();
        }
        
        function drawAvatar(lm, w, h) {
            const [nose, lSh, rSh, lEl, rEl, lWr, rWr, lHip, rHip, lKn, rKn, lAn, rAn] = [0,11,12,13,14,15,16,23,24,25,26,27,28].map(i => lm[i]);
            const shC = { x: (lSh.x + rSh.x) / 2 * w, y: (lSh.y + rSh.y) / 2 * h };
            const hiC = { x: (lHip.x + rHip.x) / 2 * w, y: (lHip.y + rHip.y) / 2 * h };
            gestureCtx.strokeStyle = colors.stem; gestureCtx.lineCap = 'round';
            gestureCtx.lineWidth = 8;
            [[lAn, lKn, lHip], [rAn, rKn, rHip]].forEach(([a, k, hp]) => {
                gestureCtx.beginPath(); gestureCtx.moveTo(a.x*w, a.y*h); gestureCtx.quadraticCurveTo(k.x*w, k.y*h, hp.x*w, hp.y*h); gestureCtx.stroke();
            });
            const mx = (hiC.x + shC.x) / 2 + Math.sin(animPhase) * 2, my = (hiC.y + shC.y) / 2;
            gestureCtx.lineWidth = 10; gestureCtx.beginPath(); gestureCtx.moveTo(hiC.x, hiC.y); gestureCtx.quadraticCurveTo(mx, my, shC.x, shC.y); gestureCtx.stroke();
            drawLeaf(mx - 8, my - 10, -0.8, 14); drawLeaf(mx + 8, my + 6, 0.8, 12);
            gestureCtx.lineWidth = 6;
            [[lSh, lEl, lWr, true], [rSh, rEl, rWr, false]].forEach(([sh, el, wr, isL]) => {
                gestureCtx.beginPath(); gestureCtx.moveTo(sh.x*w, sh.y*h); gestureCtx.lineTo(el.x*w, el.y*h); gestureCtx.lineTo(wr.x*w, wr.y*h); gestureCtx.stroke();
                drawLeaf(el.x*w, el.y*h, isL ? -0.6 : 0.6, 12);
                for (let i = -1; i <= 1; i++) drawLeaf(wr.x*w, wr.y*h, (isL ? Math.PI : 0) + i * 0.4, 10);
            });
            const hx = nose.x * w, hy = nose.y * h, r = 24;
            gestureCtx.lineWidth = 7; gestureCtx.beginPath(); gestureCtx.moveTo(shC.x, shC.y); gestureCtx.lineTo(hx, hy + r * 0.3); gestureCtx.stroke();
            for (let i = 0; i < 8; i++) {
                gestureCtx.save(); gestureCtx.translate(hx, hy); gestureCtx.rotate((i / 8) * Math.PI * 2 + animPhase * 0.04);
                gestureCtx.fillStyle = colors.flower; gestureCtx.beginPath(); gestureCtx.ellipse(0, -r * 0.4, r * 0.28, r * 0.45, 0, 0, Math.PI * 2); gestureCtx.fill(); gestureCtx.restore();
            }
            gestureCtx.fillStyle = colors.center; gestureCtx.beginPath(); gestureCtx.arc(hx, hy, r * 0.28, 0, Math.PI * 2); gestureCtx.fill();
        }
        
        function drawLeaf(x, y, ang, sz) {
            gestureCtx.save(); gestureCtx.translate(x, y); gestureCtx.rotate(ang + Math.sin(animPhase + x * 0.02) * 0.05);
            gestureCtx.fillStyle = colors.leaf; gestureCtx.beginPath(); gestureCtx.moveTo(0, 0);
            gestureCtx.quadraticCurveTo(sz * 0.5, -sz * 0.26, sz, 0); gestureCtx.quadraticCurveTo(sz * 0.5, sz * 0.26, 0, 0);
            gestureCtx.fill(); gestureCtx.restore();
        }
        
        function detectGestures(lm) {
            const now = Date.now(), lWr = lm[15], rWr = lm[16];
            handHistory.left.push({ x: lWr.x, y: lWr.y, t: now }); handHistory.right.push({ x: rWr.x, y: rWr.y, t: now });
            if (handHistory.left.length > histLen) handHistory.left.shift();
            if (handHistory.right.length > histLen) handHistory.right.shift();
            if (handHistory.left.length < 4) return;
            const lV = calcVel(handHistory.left), rV = calcVel(handHistory.right);
            const lSwR = lV.x < -swipeThreshold && Math.abs(lV.y) < Math.abs(lV.x) * 0.8;
            const lSwL = lV.x > swipeThreshold && Math.abs(lV.y) < Math.abs(lV.x) * 0.8;
            const rSwR = rV.x < -swipeThreshold && Math.abs(rV.y) < Math.abs(rV.x) * 0.8;
            const rSwL = rV.x > swipeThreshold && Math.abs(rV.y) < Math.abs(rV.x) * 0.8;
            const rStill = Math.abs(rV.x) < 0.4, lStill = Math.abs(lV.x) < 0.4;
            if (!langCooldown) {
                if ((lSwR && rStill) || (rSwR && lStill)) { cycleLang(1); langCooldown = true; handHistory = { left: [], right: [] }; setTimeout(() => langCooldown = false, 700); }
                else if ((lSwL && rStill) || (rSwL && lStill)) { cycleLang(-1); langCooldown = true; handHistory = { left: [], right: [] }; setTimeout(() => langCooldown = false, 700); }
            }
        }
        
        function calcVel(h) {
            if (h.length < 3) return { x: 0, y: 0 };
            let vx = 0, vy = 0, tw = 0;
            for (let i = 1; i < h.length; i++) {
                const dt = (h[i].t - h[i-1].t) / 1000;
                if (dt > 0 && dt < 0.5) { vx += ((h[i].x - h[i-1].x) / dt) * i; vy += ((h[i].y - h[i-1].y) / dt) * i; tw += i; }
            }
            return tw ? { x: vx / tw, y: vy / tw } : { x: 0, y: 0 };
        }
        
        function showFeedback(txt) {
            const el = document.getElementById('gestureFeedback');
            el.textContent = txt; el.classList.remove('active'); void el.offsetWidth; el.classList.add('active');
            setTimeout(() => el.classList.remove('active'), 500);
        }
        
        // Signal canvas - use getBoundingClientRect for robust sizing in fixed containers
        const sigCanvas = document.getElementById('signalCanvas'), sigCtx = sigCanvas.getContext('2d');
        let lastWaveform = null, lastWaveActive = false;
        let idleAnimFrame = null;
        
        function resizeSig() {
            const rect = sigCanvas.getBoundingClientRect();
            const w = Math.round(rect.width) || (window.innerWidth - 40);
            const h = 72; // fixed height to avoid zero-height bug
            if (sigCanvas.width !== w || sigCanvas.height !== h) {
                sigCanvas.width = w;
                sigCanvas.height = h;
            }
        }
        
        function drawSignal(data, active) {
            lastWaveform = data; lastWaveActive = active;
            if (idleAnimFrame) { cancelAnimationFrame(idleAnimFrame); idleAnimFrame = null; }
            resizeSig();
            const w = sigCanvas.width, h = sigCanvas.height;
            if (w < 10 || h < 10) return;
            sigCtx.clearRect(0, 0, w, h);
            if (!data || data.length < 2) return;
            const min = Math.min(...data), max = Math.max(...data), range = max - min || 1, pad = range * 0.1;
            sigCtx.beginPath(); sigCtx.strokeStyle = active ? '#ff6ac8' : '#1cffa8'; sigCtx.lineWidth = 2;
            for (let i = 0; i < data.length; i++) { const x = (i / data.length) * w, y = h - ((data[i] - min + pad) / (range + pad * 2)) * h; i ? sigCtx.lineTo(x, y) : sigCtx.moveTo(x, y); }
            sigCtx.stroke();
        }
        
        // Idle animation: draw a gentle sine wave when no ESP32 data
        function drawIdleWave() {
            if (lastWaveform) return; // real data arrived, stop idle
            resizeSig();
            const w = sigCanvas.width, h = sigCanvas.height, t = Date.now() / 1000;
            if (w < 10 || h < 10) { idleAnimFrame = requestAnimationFrame(drawIdleWave); return; }
            sigCtx.clearRect(0, 0, w, h);
            sigCtx.beginPath();
            sigCtx.strokeStyle = 'rgba(28, 255, 168, 0.25)';
            sigCtx.lineWidth = 1.5;
            for (let i = 0; i < w; i++) {
                const x = i;
                const y = h / 2 + Math.sin(i * 0.02 + t * 2) * (h * 0.2) + Math.sin(i * 0.005 + t * 0.7) * (h * 0.15);
                i ? sigCtx.lineTo(x, y) : sigCtx.moveTo(x, y);
            }
            sigCtx.stroke();
            idleAnimFrame = requestAnimationFrame(drawIdleWave);
        }
        
        // Redraw on resize
        function redrawSig() { if (lastWaveform) drawSignal(lastWaveform, lastWaveActive); }
        
        // Video
        const plantVideo = document.getElementById('plantVideo'); let vidState = 'nohand';
        function switchVideo(s) { if (vidState !== s) { plantVideo.src = `/videos/${s}.mp4`; plantVideo.load(); plantVideo.play().catch(() => {}); vidState = s; } }
        
        // Mode
        function updateMode() {
            document.getElementById('toggleModeBtn').textContent = currentMode === 'training' ? 'Exhibition' : 'Training';
            document.getElementById('trainingStatsBar').classList.toggle('visible', currentMode === 'training');
        }
        document.getElementById('toggleModeBtn').addEventListener('click', () => fetch('/toggle_mode', { method: 'POST' }).then(r => r.json()).then(d => { currentMode = d.mode; updateMode(); }));
        document.getElementById('retrainBtn').addEventListener('click', () => fetch('/retrain', { method: 'POST' }));
        
        // Socket
        socket.on('update', d => {
            document.getElementById('connectionDot').classList.toggle('connected', d.connected);
            currentMode = d.mode; updateMode();
            document.getElementById('touchCount').textContent = d.touch_count;
            document.getElementById('restCount').textContent = d.notouch_count;
            document.getElementById('accuracy').textContent = d.model_accuracy > 0 ? `${(d.model_accuracy * 100).toFixed(0)}%` : '--';
            document.getElementById('trainingProgressFill').style.width = `${(d.warmup_progress || 0) * 100}%`;
            
            const vc = document.getElementById('videoContainer'), resp = document.getElementById('plantResponse');
            const needle = document.getElementById('confidenceNeedle'), cVal = document.getElementById('confidenceValue');
            
            if (currentMode === 'training') {
                const phase = d.training_phase;
                if (phase === 'touch') {
                    vc.classList.add('ping-active'); vc.classList.remove('touch-active');
                    resp.textContent = t('touchNow'); resp.className = 'plant-response ping-active';
                } else if (phase === 'release' || phase === 'notouch') {
                    vc.classList.remove('ping-active'); vc.classList.remove('touch-active');
                    resp.textContent = t('letGo'); resp.className = 'plant-response resting';
                } else {
                    vc.classList.remove('ping-active'); vc.classList.remove('touch-active');
                    resp.textContent = t('resting'); resp.className = 'plant-response resting';
                }
                lastPingActive = d.ping_active;
                needle.style.left = '50%'; cVal.textContent = '--';
            } else {
                vc.classList.remove('ping-active'); resp.classList.remove('ping-active');
                if (d.prediction === 'touch') {
                    vc.classList.add('touch-active'); switchVideo('hand');
                    resp.textContent = t('touch'); resp.className = 'plant-response touch';
                    const pos = d.confidence > 0.85 ? 0.06 : d.confidence > 0.7 ? 0.18 : d.confidence > 0.5 ? 0.32 : 0.45;
                    needle.style.left = (pos * 100) + '%';
                } else {
                    vc.classList.remove('touch-active'); switchVideo('nohand');
                    resp.textContent = t('resting'); resp.className = 'plant-response resting';
                    const pos = d.confidence > 0.85 ? 0.94 : d.confidence > 0.7 ? 0.82 : d.confidence > 0.5 ? 0.68 : 0.55;
                    needle.style.left = (pos * 100) + '%';
                }
                cVal.textContent = d.confidence > 0 ? (d.confidence * 100).toFixed(0) + '%' : '--';
            }
            
            if (d.waveform?.length > 0) drawSignal(d.waveform, (currentMode === 'training' && d.training_phase === 'touch') || (currentMode === 'exhibition' && d.prediction === 'touch'));
        });
        
        // QR - small
        if (window.QRCode) {
            new QRCode(document.getElementById('qrBox'), {
                text: plantWs,
                width: 60,
                height: 60,
                colorDark: "#1cffa8",
                colorLight: "#0a0f14",
                correctLevel: QRCode.CorrectLevel.M
            });
            document.getElementById('qrLabel').textContent = plantWs;
        }

        window.addEventListener('resize', () => { resizeSig(); redrawSig(); });
        // Delay initial sizing to let fixed containers settle
        requestAnimationFrame(() => { resizeSig(); drawIdleWave(); });
        setTimeout(() => { resizeSig(); if (!lastWaveform) drawIdleWave(); else redrawSig(); }, 200);
        updateMode(); initGesture(); updateConfLabels();
        document.addEventListener('click', () => plantVideo.play().catch(() => {}), { once: true });
    </script>
</body>
</html>
'''




@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, plant_ws=f"ws://{ESP32_IP}:{ESP32_PORT}")


@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)


@app.route('/fonts/<path:filename>')
def serve_font(filename):
    """Serve font files with case-insensitive recursive search across all font dirs."""
    target = filename.lower()
    # Search in all possible font locations
    search_dirs = [FONTS_DIR, FONTS_BASE]
    for base in search_dirs:
        if base.exists():
            for font_file in base.rglob('*'):
                if font_file.is_file() and font_file.name.lower() == target:
                    return send_from_directory(font_file.parent, font_file.name)
            # Also try matching without extension variations
            stem = Path(target).stem
            for font_file in base.rglob('*'):
                if font_file.is_file() and font_file.stem.lower().replace(' ', '') == stem.replace(' ', ''):
                    return send_from_directory(font_file.parent, font_file.name)
    print(f"[FONT] NOT FOUND: {filename} (searched in {FONTS_DIR}, {FONTS_BASE})")
    return f"Font not found: {filename}", 404


@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    if state.mode == 'training' and state.model.is_trained:
        state.mode = 'exhibition'
    else:
        state.mode = 'training'
        state.last_ping_time = time.time()
        state.next_ping_interval = random.uniform(PING_MIN_INTERVAL, PING_MAX_INTERVAL)
        state.ping_active = False
    return jsonify({'mode': state.mode})


@app.route('/retrain', methods=['POST'])
def retrain():
    state.mode = 'training'
    state.last_ping_time = time.time()
    state.next_ping_interval = random.uniform(PING_MIN_INTERVAL, PING_MAX_INTERVAL)
    state.ping_active = False
    state.model.force_retrain()
    return jsonify({'status': 'ok'})




async def esp32_listener():
    while True:
        try:
            async with websockets.connect(f"ws://{ESP32_IP}:{ESP32_PORT}", ping_interval=20) as ws:
                state.connected = True
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)
                        if data["type"] == "data":
                            for v in data["voltages"]:
                                state.voltage_buffer.append(v)
                            if data["voltages"]:
                                state.current_voltage = data["voltages"][-1]
                            process_voltage_for_training(data["voltages"], time.time())
                            
                            if state.mode == 'exhibition' and state.model.is_trained and len(state.voltage_buffer) >= WINDOW_SIZE:
                                if time.time() - state.last_prediction_time >= PREDICTION_INTERVAL:
                                    samples = [v / 1000.0 for v in list(state.voltage_buffer)[-WINDOW_SIZE:]]
                                    features = extract_features(np.array(samples))
                                    if features:
                                        pred, conf = state.model.predict(features)
                                        if pred is not None:
                                            state.prediction = "touch" if pred == 1 else "resting"
                                            state.confidence = conf
                                    state.last_prediction_time = time.time()
                    except asyncio.TimeoutError:
                        pass
        except Exception as e:
            state.connected = False
            await asyncio.sleep(2)




def run_esp32():
    asyncio.new_event_loop().run_until_complete(esp32_listener())




def broadcast():
    while True:
        # Determine training phase for UI feedback
        if state.ping_active:
            t_elapsed = time.time() - state.ping_start_time
            if t_elapsed < POST_PING_CAPTURE_END:
                training_phase = 'touch'       # 0-5s: visitor should touch
            elif t_elapsed < NOTOUCH_CAPTURE_START:
                training_phase = 'release'     # 5-7s: let go
            elif t_elapsed < NOTOUCH_CAPTURE_END:
                training_phase = 'notouch'     # 7-12s: collecting baseline
            else:
                training_phase = 'waiting'
        else:
            training_phase = 'waiting'         # between pings
        
        socketio.emit('update', {
            'connected': state.connected, 'mode': state.mode,
            'waveform': list(state.voltage_buffer)[-500:] if state.voltage_buffer else [],
            'prediction': state.prediction, 'confidence': state.confidence,
            'ping_active': state.ping_active,
            'training_phase': training_phase,
            'touch_count': state.touch_count, 'notouch_count': state.notouch_count,
            'warmup_progress': state.model.warmup_progress,
            'model_accuracy': state.model.accuracy_estimate,
            'is_trained': state.model.is_trained,
        })
        time.sleep(UPDATE_RATE_MS / 1000)




if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 PHAENOMENA - BERÜHRUNGSWAHRNEHMUNG")
    print("   Unified Exhibition Interface with Gesture Controls")
    print("="*60)
    print(f"🌐 http://localhost:5004")
    
    # Font diagnostics
    print(f"\n📝 Font directory: {FONTS_DIR.resolve()}")
    print(f"📝 Font base: {FONTS_BASE.resolve()}")
    for fdir in [FONTS_DIR, FONTS_BASE]:
        if fdir.exists():
            font_files = list(fdir.rglob('*.otf')) + list(fdir.rglob('*.ttf')) + list(fdir.rglob('*.woff*'))
            print(f"   Found {len(font_files)} font files in {fdir}:")
            for f in sorted(font_files):
                print(f"     → {f.relative_to(fdir)}")
        else:
            print(f"   ⚠️  Directory does not exist: {fdir.resolve()}")
    
    print("="*60 + "\n")
    
    loaded, age = state.model.load_latest()
    if loaded:
        state.mode = 'exhibition'
        print(f"✨ Model loaded (age: {age:.0f} min) - exhibition mode")
    else:
        print("🔧 No model found - training mode")
        state.last_ping_time = time.time()
        state.next_ping_interval = random.uniform(PING_MIN_INTERVAL, PING_MAX_INTERVAL)
    
    threading.Thread(target=run_esp32, daemon=True).start()
    threading.Thread(target=broadcast, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5004, debug=False, allow_unsafe_werkzeug=True)
