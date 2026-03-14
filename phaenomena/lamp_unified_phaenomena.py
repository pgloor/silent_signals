#!/usr/bin/env python3
"""
🌿 Phaenomena - LICHTWAHRNEHMUNG (Light Perception)
Unified Exhibition Interface with Gesture Controls

Features:
- Phänomena Corporate Design (Town & Country title, Satoshi body)
- Gesture control for language selection (swipe left/right)
- Corner overlay camera for gesture detection
- Two-panel layout matching CD guidelines
- 4-language support (DE/FR/IT/EN)

Port: 5002
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
import wave

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

from flask import Flask, render_template_string, jsonify, send_from_directory, request
from flask_socketio import SocketIO
import websockets
import requests

# ===== CONFIGURATION =====
ESP32_IP = "192.168.1.134"
ESP32_PORT = 81
SHELLY_IP = "192.168.1.130"

SAMPLE_RATE = 100
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

WARMUP_PAIRS = 20
MAX_TRAINING_PAIRS = 100
RETRAIN_INTERVAL = 8
MIN_PAIRS_FOR_RETRAIN = 15

LAMP_ON_DURATION = 15
LAMP_OFF_DURATION = 15
POST_TOGGLE_DELAY = 3

EXHIBITION_DETECTION_WINDOW = 5
MODEL_MAX_AGE_MINUTES = 30

UPDATE_RATE_MS = 200
PREDICTION_INTERVAL = 0.5

OUTPUT_DIR = Path("./lamp_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_DIR = Path("./plant-viz-movies")
FONTS_DIR = Path("./fonts/Fonts")


def save_wav_file(samples_v, label, sample_number):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plant_{label}_n{sample_number:03d}_{timestamp}.wav"
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


# ===== SHELLY PLUG CONTROLLER =====
class ShellyPlug:
    def __init__(self, ip):
        self.ip = ip
        self.connected = False
        self.last_status = None
        self._check_connection()
    
    def _check_connection(self):
        try:
            r = requests.get(f"http://{self.ip}/rpc/Switch.GetStatus?id=0", timeout=3)
            if r.status_code == 200:
                self.connected = True
                self.last_status = r.json().get('output', False)
                print(f"✅ Shelly plug connected: {self.ip}")
        except:
            self.connected = False
    
    def get_status(self):
        try:
            r = requests.get(f"http://{self.ip}/rpc/Switch.GetStatus?id=0", timeout=3)
            if r.status_code == 200:
                self.last_status = r.json().get('output', False)
                self.connected = True
                return self.last_status
        except:
            pass
        return None
    
    def turn_on(self):
        try:
            r = requests.get(f"http://{self.ip}/rpc/Switch.Set?id=0&on=true", timeout=3)
            if r.status_code == 200:
                self.last_status = True
                return True
        except:
            pass
        return False
    
    def turn_off(self):
        try:
            r = requests.get(f"http://{self.ip}/rpc/Switch.Set?id=0&on=false", timeout=3)
            if r.status_code == 200:
                self.last_status = False
                return True
        except:
            pass
        return False


# ===== FEATURE EXTRACTION =====
def extract_features(samples, sr=SAMPLE_RATE):
    features = {}
    samples = np.array(samples)
    n = len(samples)
    
    if n < 100:
        return None
    
    samples = samples.astype(np.float64)
    eps = 1e-10
    
    features['mean_voltage'] = float(np.mean(samples))
    features['median_voltage'] = float(np.median(samples))
    features['voltage_std'] = float(np.std(samples) + eps)
    features['voltage_min'] = float(np.min(samples))
    features['voltage_max'] = float(np.max(samples))
    features['voltage_range'] = float(np.ptp(samples))
    features['voltage_iqr'] = float(np.percentile(samples, 75) - np.percentile(samples, 25))
    
    n_segments = 10
    segment_len = n // n_segments
    segment_means = []
    segment_stds = []
    
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len
        seg = samples[start:end]
        segment_means.append(np.mean(seg))
        segment_stds.append(np.std(seg))
    
    segment_means = np.array(segment_means)
    segment_stds = np.array(segment_stds)
    
    features['trend_start_to_end'] = float(segment_means[-1] - segment_means[0])
    features['trend_magnitude'] = float(np.abs(segment_means[-1] - segment_means[0]))
    features['first_half_mean'] = float(np.mean(segment_means[:5]))
    features['second_half_mean'] = float(np.mean(segment_means[5:]))
    features['half_difference'] = float(features['second_half_mean'] - features['first_half_mean'])
    features['segment_mean_std'] = float(np.std(segment_means))
    features['segment_mean_range'] = float(np.ptp(segment_means))
    
    diffs = np.diff(segment_means)
    features['monotonic_score'] = float(np.abs(np.sum(np.sign(diffs))) / len(diffs))
    features['rising_segments'] = float(np.sum(diffs > 0) / len(diffs))
    
    features['overall_variability'] = float(np.std(samples) + eps)
    features['mad'] = float(np.median(np.abs(samples - np.median(samples))) + eps)
    features['variability_trend'] = float(segment_stds[-1] - segment_stds[0])
    features['variability_mean'] = float(np.mean(segment_stds))
    mean_abs = np.abs(features['mean_voltage']) + eps
    features['coeff_variation'] = float(features['voltage_std'] / mean_abs)
    
    diff1 = np.diff(samples)
    features['mean_rate_of_change'] = float(np.mean(diff1))
    features['abs_rate_of_change'] = float(np.mean(np.abs(diff1)))
    features['rate_std'] = float(np.std(diff1) + eps)
    
    diff2 = np.diff(diff1)
    features['smoothness'] = float(1.0 / (np.std(diff2) + eps))
    features['jitter'] = float(np.mean(np.abs(diff2)))
    
    sign_changes = np.sum(np.abs(np.diff(np.sign(diff1))) > 0)
    features['direction_changes'] = float(sign_changes)
    features['direction_change_rate'] = float(sign_changes / n)
    
    try:
        peaks, _ = find_peaks(samples, distance=sr//10)
        valleys, _ = find_peaks(-samples, distance=sr//10)
        features['n_peaks'] = float(len(peaks))
        features['n_valleys'] = float(len(valleys))
        features['n_extrema'] = float(len(peaks) + len(valleys))
        features['extrema_rate'] = float((len(peaks) + len(valleys)) / (n / sr))
    except:
        features['n_peaks'] = 0
        features['n_valleys'] = 0
        features['n_extrema'] = 0
        features['extrema_rate'] = 0
    
    fft_vals = np.abs(fft(samples))[:n//2]
    freqs = fftfreq(n, 1/sr)[:n//2]
    
    bio_bands = [
        ('ultra_slow', 0.05, 0.2),
        ('slow', 0.2, 0.5),
        ('mid_slow', 0.5, 1.0),
        ('mid', 1.0, 2.0),
        ('fast', 2.0, 5.0),
    ]
    
    total_power = np.sum(fft_vals**2) + eps
    
    for name, low, high in bio_bands:
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_power = np.sum(fft_vals[mask]**2)
            features[f'power_{name}'] = float(band_power)
            features[f'ratio_{name}'] = float(band_power / total_power)
        else:
            features[f'power_{name}'] = 0.0
            features[f'ratio_{name}'] = 0.0
    
    window = sr // 2
    if window > 0 and n > window:
        n_windows = n // window
        upper_env = []
        lower_env = []
        for i in range(n_windows):
            start = i * window
            end = start + window
            seg = samples[start:end]
            upper_env.append(np.max(seg))
            lower_env.append(np.min(seg))
        upper_env = np.array(upper_env)
        lower_env = np.array(lower_env)
        features['envelope_width_mean'] = float(np.mean(upper_env - lower_env))
        features['envelope_width_trend'] = float((upper_env[-1] - lower_env[-1]) - (upper_env[0] - lower_env[0]))
    else:
        features['envelope_width_mean'] = 0.0
        features['envelope_width_trend'] = 0.0
    
    centered = samples - np.mean(samples)
    std = np.std(centered) + eps
    features['skewness'] = float(np.mean((centered / std) ** 3))
    features['kurtosis'] = float(np.mean((centered / std) ** 4) - 3)
    
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + eps)
    
    half_second = sr // 2
    if len(autocorr) > half_second:
        features['autocorr_decay'] = float(autocorr[half_second])
    else:
        features['autocorr_decay'] = 0.0
    
    features['rms'] = float(np.sqrt(np.mean(samples**2)))
    features['energy'] = float(np.sum(samples**2))
    
    return features


_test_features = extract_features(np.zeros(WINDOW_SIZE), SAMPLE_RATE)
FEATURE_NAMES = list(_test_features.keys()) if _test_features else []


def features_to_array(features):
    if features is None:
        return None
    return np.array([features.get(name, 0) for name in FEATURE_NAMES])


# ===== ADAPTIVE MODEL =====
class AdaptiveModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = deque(maxlen=MAX_TRAINING_PAIRS)
        self.is_trained = False
        self.training_count = 0
        self.last_retrain_count = 0
        self.accuracy_estimate = 0
        self.lock = threading.Lock()
        
    def add_sample(self, features, label):
        if features is None:
            return
        
        feature_array = features_to_array(features)
        if feature_array is None:
            return
            
        with self.lock:
            self.training_data.append((feature_array, label))
            self.training_count += 1
            
            pairs_since_retrain = self.training_count - self.last_retrain_count
            total_pairs = len(self.training_data)
            
            should_retrain = (
                total_pairs >= MIN_PAIRS_FOR_RETRAIN and 
                (pairs_since_retrain >= RETRAIN_INTERVAL or not self.is_trained)
            )
            
            if should_retrain:
                self._retrain()
    
    def force_retrain(self):
        with self.lock:
            self._retrain()
    
    def _retrain(self):
        if len(self.training_data) < MIN_PAIRS_FOR_RETRAIN:
            return
        
        X = np.array([d[0] for d in self.training_data])
        y = np.array([d[1] for d in self.training_data])
        
        n_on = np.sum(y == 1)
        n_off = np.sum(y == 0)
        
        if n_on < 3 or n_off < 3:
            return
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        predictions = self.model.predict(X_scaled)
        self.accuracy_estimate = np.mean(predictions == y)
        
        self.is_trained = True
        self.last_retrain_count = self.training_count
        
        print(f"🔄 Model retrained: {len(self.training_data)} samples, {self.accuracy_estimate:.1%} accuracy")
        self._save_snapshot()
    
    def predict(self, features):
        with self.lock:
            if not self.is_trained or self.model is None:
                return None, 0
            
            X = features_to_array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = max(proba)
            
            return prediction, confidence
    
    def _save_snapshot(self):
        snapshot = {
            'model': self.model, 'scaler': self.scaler, 'feature_names': FEATURE_NAMES,
            'training_count': self.training_count, 'last_retrain_count': self.last_retrain_count,
            'training_data': list(self.training_data), 'accuracy_estimate': self.accuracy_estimate,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        with open(OUTPUT_DIR / "lamp_model_latest.pkl", 'wb') as f:
            pickle.dump(snapshot, f)
    
    def load_latest(self):
        latest_path = OUTPUT_DIR / "lamp_model_latest.pkl"
        if latest_path.exists():
            try:
                age_minutes = (time.time() - latest_path.stat().st_mtime) / 60
                with open(latest_path, 'rb') as f:
                    snapshot = pickle.load(f)
                self.model = snapshot['model']
                self.scaler = snapshot['scaler']
                self.training_count = snapshot.get('training_count', 0)
                self.last_retrain_count = snapshot.get('last_retrain_count', self.training_count)
                self.accuracy_estimate = snapshot.get('accuracy_estimate', 0)
                self.is_trained = True
                if 'training_data' in snapshot:
                    self.training_data = deque(snapshot['training_data'], maxlen=MAX_TRAINING_PAIRS)
                print(f"✨ Model loaded ({len(self.training_data)} samples)")
                return True, age_minutes
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        return False, 0
    
    @property
    def warmup_progress(self):
        return min(1.0, len(self.training_data) / WARMUP_PAIRS)


# ===== APPLICATION STATE =====
class AppState:
    def __init__(self):
        self.voltage_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.current_voltage = 0
        self.connected = False
        self.plug_connected = False
        self.mode = 'training'
        
        self.lamp_on = False
        self.lamp_toggle_time = None
        self.next_toggle_time = None
        
        self.training_phase = 'idle'
        self.collection_samples = []
        self.lamp_on_count = 0
        self.lamp_off_count = 0
        
        self.prediction = 'lamp_off'
        self.confidence = 0
        self.last_prediction_time = 0
        
        self.model = AdaptiveModel()

state = AppState()
shelly_plug = None


def process_voltage_for_training(voltages, current_time):
    voltages_v = [v / 1000.0 for v in voltages]
    
    if state.training_phase == 'waiting':
        time_since_toggle = current_time - state.lamp_toggle_time if state.lamp_toggle_time else 0
        if time_since_toggle >= POST_TOGGLE_DELAY:
            state.training_phase = 'collecting_on' if state.lamp_on else 'collecting_off'
            state.collection_samples = []
            print(f"  📼 Collecting {'lamp-on' if state.lamp_on else 'lamp-off'} sample...")
    
    if state.training_phase in ['collecting_on', 'collecting_off']:
        state.collection_samples.extend(voltages_v)
        
        if len(state.collection_samples) >= WINDOW_SIZE:
            samples_array = np.array(state.collection_samples[:WINDOW_SIZE])
            label = 1 if state.training_phase == 'collecting_on' else 0
            label_str = "lamp_on" if label == 1 else "lamp_off"
            
            sample_num = state.lamp_on_count if label == 1 else state.lamp_off_count
            save_wav_file(samples_array, label_str, sample_num)
            
            features = extract_features(samples_array, SAMPLE_RATE)
            if features is not None:
                state.model.add_sample(features, label)
                if label == 1:
                    state.lamp_on_count += 1
                else:
                    state.lamp_off_count += 1
                print(f"  ✅ Sample added: {label_str}")
            
            state.training_phase = 'idle'
            state.collection_samples = []


def training_lamp_controller():
    while True:
        if state.mode == 'training' and shelly_plug and shelly_plug.connected:
            current_time = time.time()
            
            if state.next_toggle_time and current_time >= state.next_toggle_time:
                new_state = not state.lamp_on
                success = shelly_plug.turn_on() if new_state else shelly_plug.turn_off()
                
                if success:
                    state.lamp_on = new_state
                    state.lamp_toggle_time = current_time
                    state.training_phase = 'waiting'
                    duration = LAMP_ON_DURATION if new_state else LAMP_OFF_DURATION
                    state.next_toggle_time = current_time + duration
                    print(f"💡 Auto-toggle: Lamp {'ON' if new_state else 'OFF'}")
        
        time.sleep(0.5)


async def esp32_listener():
    uri = f"ws://{ESP32_IP}:{ESP32_PORT}"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                state.connected = True
                print("✅ Connected to ESP32")
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(message)
                        current_time = time.time()
                        
                        if data.get("type") == "data":
                            voltages = data.get("voltages", [])
                            for v in voltages:
                                state.voltage_buffer.append(v)
                            if voltages:
                                state.current_voltage = voltages[-1]
                            
                            process_voltage_for_training(voltages, current_time)
                            
                            if (state.model.is_trained and 
                                len(state.voltage_buffer) >= WINDOW_SIZE and
                                current_time - state.last_prediction_time >= PREDICTION_INTERVAL):
                                
                                samples = list(state.voltage_buffer)[-WINDOW_SIZE:]
                                samples_v = [v / 1000.0 for v in samples]
                                features = extract_features(np.array(samples_v), SAMPLE_RATE)
                                if features:
                                    pred, conf = state.model.predict(features)
                                    if pred is not None:
                                        state.prediction = "lamp_on" if pred == 1 else "lamp_off"
                                        state.confidence = conf
                                state.last_prediction_time = current_time
                    except asyncio.TimeoutError:
                        pass
        except Exception as e:
            state.connected = False
            print(f"❌ ESP32 error: {e}")
            await asyncio.sleep(2)


def run_esp32_listener():
    asyncio.new_event_loop().run_until_complete(esp32_listener())


def broadcast_updates():
    while True:
        waveform = list(state.voltage_buffer)[-500:] if state.voltage_buffer else []
        collecting = state.training_phase in ['collecting_on', 'collecting_off']
        
        socketio.emit('update', {
            'connected': state.connected,
            'plug_connected': state.plug_connected,
            'mode': state.mode,
            'waveform': waveform,
            'voltage': state.current_voltage,
            'lamp_on': state.lamp_on,
            'prediction': state.prediction,
            'confidence': state.confidence,
            'collecting': collecting,
            'lamp_on_count': state.lamp_on_count,
            'lamp_off_count': state.lamp_off_count,
            'warmup_progress': state.model.warmup_progress,
            'model_accuracy': state.model.accuracy_estimate,
            'is_trained': state.model.is_trained,
        })
        time.sleep(UPDATE_RATE_MS / 1000)


# ===== FLASK APP =====
app = Flask(__name__)
app.config['SECRET_KEY'] = 'phaenomena-lamp-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LICHTWAHRNEHMUNG - Phänomena</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <style>
        @font-face { font-family: 'TownCountry'; font-display: swap;
            src: url('/fonts/TownandCountryJNL.otf') format('opentype'),
                 url('/fonts/TownandCountryJNL-Regular.otf') format('opentype'),
                 url('/fonts/TownandCountryJNL.ttf') format('truetype'); }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Light.otf') format('opentype'); font-weight: 300; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Regular.otf') format('opentype'); font-weight: 400; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Medium.otf') format('opentype'); font-weight: 500; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Bold.otf') format('opentype'); font-weight: 700; font-display: swap; }
        @font-face { font-family: 'Satoshi'; src: url('/fonts/Satoshi-Black.otf') format('opentype'); font-weight: 900; font-display: swap; }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --dark: #201844; --green: #1cffa8; --cyan: #2fd6ee; --yellow: #fbbf24; --panel-bg: rgba(40,30,80,0.6); }
        
        body { background: var(--dark); color: #fff; font-family: 'Satoshi', sans-serif; height: 100vh; overflow: hidden; }
        
        .page-layout { display: flex; flex-direction: column; height: 100vh; padding: 0; }
        .content-area { display: flex; flex-direction: column; flex: 1; padding: 30px 40px 0 40px; min-height: 0; }
        
        .header { text-align: center; margin-bottom: 30px; }
        .exhibit-title { font-family: 'TownCountry', serif; font-size: 3.2em; font-weight: 400; letter-spacing: 0.06em; color: var(--green); margin-bottom: 10px; }
        .exhibit-subtitle { font-size: 1.4em; font-weight: 400; color: rgba(255,255,255,0.85); }
        
        .connection-dot { position: fixed; top: 20px; right: 20px; width: 12px; height: 12px; border-radius: 50%; background: #ef4444; z-index: 100; }
        .connection-dot.connected { background: var(--green); box-shadow: 0 0 12px var(--green); }
        .connection-dot.plug { right: 40px; background: var(--yellow); }
        .connection-dot.plug.connected { background: var(--yellow); box-shadow: 0 0 12px var(--yellow); }
        
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; flex: 1; min-height: 0; }
        
        .panel { background: var(--panel-bg); border-radius: 16px; padding: 24px; display: flex; flex-direction: column; align-items: center; }
        .panel-title { font-family: 'TownCountry', serif; font-size: 1.2em; font-weight: 400; letter-spacing: 0.04em; color: rgba(255,255,255,0.85); margin-bottom: 18px; text-align: center; }
        
        .lamp-content { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px; width: 100%; }
        .lamp-icon { font-size: 6em; transition: all 0.3s; filter: grayscale(1) opacity(0.4); }
        .lamp-icon.on { filter: none; text-shadow: 0 0 60px var(--yellow); }
        .lamp-status { font-size: 1.4em; font-weight: 600; color: rgba(255,255,255,0.5); }
        .lamp-status.on { color: var(--yellow); }
        
        .lamp-buttons { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
        .lamp-btn { padding: 14px 28px; font-size: 1.1em; font-weight: 700; border: none; border-radius: 30px; cursor: pointer; transition: all 0.3s; font-family: 'Satoshi', sans-serif; letter-spacing: 1px; }
        .lamp-btn:hover { transform: scale(1.05); }
        .lamp-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
        .lamp-btn-on { background: linear-gradient(135deg, #d97706, var(--yellow)); color: #000; box-shadow: 0 0 20px rgba(251,191,36,0.3); }
        .lamp-btn-on:hover:not(:disabled) { box-shadow: 0 0 40px rgba(251,191,36,0.5); }
        .lamp-btn-off { background: linear-gradient(135deg, #1e3a5f, #2d5a8e); color: #fff; box-shadow: 0 0 20px rgba(45,90,142,0.3); }
        .lamp-btn-off:hover:not(:disabled) { box-shadow: 0 0 40px rgba(45,90,142,0.5); }
        
        .plant-content { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; gap: 12px; width: 100%; }
        .video-container { width: 300px; height: 300px; border-radius: 50%; overflow: hidden; border: 4px solid var(--green); box-shadow: 0 0 40px rgba(28,255,168,0.3); transition: all 0.4s; flex-shrink: 0; }
        .video-container.lamp-on { border-color: var(--yellow); box-shadow: 0 0 60px rgba(251,191,36,0.5); }
        .video-container.collecting { animation: pulse-collect 1s infinite; }
        @keyframes pulse-collect { 0%,100% { transform: scale(1); } 50% { transform: scale(1.03); } }
        .video-container video { width: 100%; height: 100%; object-fit: cover; transform: scale(1.15); }
        
        .plant-sensing { font-size: 1.15em; color: rgba(255,255,255,0.7); margin-top: 12px; }
        .plant-response { font-size: 2em; font-weight: 700; text-align: center; }
        .plant-response.lamp-off { color: var(--green); }
        .plant-response.lamp-on { color: var(--yellow); text-shadow: 0 0 20px var(--yellow); }
        
        .confidence-scale { width: 100%; max-width: 360px; margin-top: 18px; }
        .scale-labels { display: flex; justify-content: space-between; font-size: 0.8em; color: rgba(255,255,255,0.7); margin-bottom: 8px; }
        .scale-labels span { text-align: center; flex: 1; line-height: 1.3; }
        .scale-bar { height: 10px; background: linear-gradient(90deg, var(--green) 0%, rgba(255,255,255,0.15) 50%, var(--yellow) 100%); border-radius: 5px; position: relative; }
        .scale-marker { position: absolute; top: -5px; width: 6px; height: 20px; background: white; border-radius: 3px; transform: translateX(-50%); transition: left 0.3s; box-shadow: 0 0 10px rgba(255,255,255,0.9); }
        .scale-ticks { display: flex; justify-content: space-between; margin-top: 5px; padding: 0 2px; }
        .scale-ticks span { width: 1px; height: 8px; background: rgba(255,255,255,0.3); }
        
        .middle-section { padding: 6px 40px; display: flex; flex-direction: column; gap: 8px; }
        .training-stats { display: none; justify-content: center; gap: 30px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 12px; }
        .training-stats.visible { display: flex; }
        .stat-item { text-align: center; }
        .stat-value { font-size: 1.3em; font-weight: 700; }
        .stat-value.on { color: var(--yellow); }
        .stat-value.off { color: var(--green); }
        .stat-label { font-size: 0.7em; color: rgba(255,255,255,0.5); }
        .training-progress { display: flex; align-items: center; gap: 8px; }
        .training-progress .progress-bar { width: 80px; height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
        .training-progress .progress-fill { height: 100%; background: var(--green); }
        
        .signal-container { width: 100%; height: 80px; background: rgba(255,255,255,0.03); border-top: 1px solid rgba(28,255,168,0.15); border-bottom: 1px solid rgba(28,255,168,0.15); padding: 4px 0; position: relative; }
        .signal-title { position: absolute; top: 6px; left: 16px; font-size: 0.65em; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1.5px; z-index: 2; }
        #signalCanvas { width: 100%; height: 100%; display: block; }
        
        .bottom-area { display: flex; align-items: flex-start; justify-content: flex-start; padding: 10px 40px 12px 40px; position: relative; min-height: 160px; }
        .language-zone { display: flex; flex-direction: column; align-items: center; gap: 8px; }
        .gesture-overlay { width: 280px; height: 190px; border-radius: 12px; overflow: hidden; border: 2px solid rgba(28,255,168,0.3); background: rgba(0,0,0,0.5); transition: all 0.3s; }
        .gesture-overlay:hover { border-color: var(--green); }
        .gesture-overlay.detected { border-color: var(--green); box-shadow: 0 0 15px rgba(28,255,168,0.3); }
        #gestureVideo { display: none; }
        #gestureCanvas { width: 100%; height: 100%; }
        .gesture-feedback { text-align: center; font-size: 1.3em; font-weight: 700; color: var(--green); text-shadow: 0 0 15px var(--green); opacity: 0; pointer-events: none; height: 0; overflow: visible; }
        .gesture-feedback.active { opacity: 1; animation: pop 0.5s ease-out; }
        @keyframes pop { 0% { transform: scale(0.5); opacity: 0; } 50% { transform: scale(1.1); opacity: 1; } 100% { transform: scale(1); opacity: 1; } }
        .swipe-hint { text-align: center; font-size: 0.7em; color: rgba(255,255,255,0.35); margin-top: 2px; }
        .language-buttons { display: flex; gap: 8px; }
        .lang-btn { padding: 10px 20px; border-radius: 18px; font-size: 1em; font-weight: 600; background: rgba(32,24,68,0.9); border: 2px solid rgba(255,255,255,0.2); color: rgba(255,255,255,0.5); cursor: pointer; transition: all 0.3s; font-family: 'Satoshi', sans-serif; min-width: 58px; text-align: center; }
        .lang-btn:hover { border-color: rgba(255,255,255,0.4); color: rgba(255,255,255,0.8); }
        .lang-btn.active { background: var(--green); border-color: var(--green); color: var(--dark); }
        
        .qr-panel { position: fixed; bottom: 14px; right: 14px; padding: 8px; border-radius: 10px; background: rgba(0,0,0,0.4); border: 1px solid rgba(28,255,168,0.25); width: 110px; z-index: 110; }
        .qr-panel h4 { margin: 0 0 4px; font-size: 0.65em; color: var(--green); font-family: 'Satoshi', sans-serif; }
        .qr-label { margin-top: 3px; font-size: 0.45em; color: rgba(255,255,255,0.6); word-break: break-all; }
        .qr-hint { margin-top: 2px; font-size: 0.45em; color: rgba(255,255,255,0.4); }
        
        .staff-controls { position: fixed; bottom: 10px; right: 140px; display: flex; gap: 4px; opacity: 0.15; transition: opacity 0.3s; z-index: 100; }
        .staff-controls:hover { opacity: 1; }
        .staff-btn { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: rgba(255,255,255,0.7); padding: 5px 10px; border-radius: 5px; cursor: pointer; font-family: inherit; font-size: 0.65em; }
        .staff-btn:hover { background: rgba(255,255,255,0.2); }
        .staff-btn.active { background: rgba(28,255,168,0.3); border-color: var(--green); color: var(--green); }
        .no-camera { display: flex; align-items: center; justify-content: center; height: 100%; font-size: 0.65em; color: rgba(255,255,255,0.4); }
    </style>
</head>
<body>
    <div class="connection-dot" id="plantDot"></div>
    <div class="connection-dot plug" id="plugDot"></div>
    
    <div class="page-layout">
        <div class="content-area">
            <div class="header">
                <div class="exhibit-title" id="exhibitTitle">LICHTWAHRNEHMUNG</div>
                <div class="exhibit-subtitle" id="exhibitSubtitle">Spüre ich Licht? Probiere es aus.</div>
            </div>
            
            <div class="main-grid">
                <div class="panel">
                    <div class="panel-title" id="lampPanelTitle">Die Lampe:</div>
                    <div class="lamp-content">
                        <div class="lamp-icon" id="lampIcon">💡</div>
                        <div class="lamp-status" id="lampStatus">Aus</div>
                        <div class="lamp-buttons">
                            <button class="lamp-btn lamp-btn-on" id="btnLampOn" onclick="lampOn()">Licht einschalten</button>
                            <button class="lamp-btn lamp-btn-off" id="btnLampOff" onclick="lampOff()">Licht ausschalten</button>
                        </div>
                    </div>
                </div>
                
                <div class="panel">
                    <div class="panel-title" id="plantPanelTitle">Das spürt die Pflanze:</div>
                    <div class="plant-content">
                        <div class="video-container" id="videoContainer">
                            <video id="plantVideo" autoplay loop muted playsinline>
                                <source src="/videos/dark.mp4" type="video/mp4">
                            </video>
                        </div>
                        <div class="plant-sensing" id="plantSensing">Ich spüre...</div>
                        <div class="plant-response lamp-off" id="plantResponse">Dunkelheit</div>
                        
                        <div class="confidence-scale">
                            <div class="scale-labels">
                                <span id="scaleLabel1">Dunkel<br>sicher</span>
                                <span id="scaleLabel2">Dunkel<br>wahrscheinlich</span>
                                <span id="scaleLabel3">Licht<br>wahrscheinlich</span>
                                <span id="scaleLabel4">Licht<br>sicher</span>
                            </div>
                            <div class="scale-bar">
                                <div class="scale-marker" id="scaleMarker" style="left:25%"></div>
                            </div>
                            <div class="scale-ticks"><span></span><span></span><span></span><span></span><span></span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="middle-section">
            <div class="training-stats" id="trainingStats">
                <div class="stat-item"><div class="stat-value on" id="lampOnCount">0</div><div class="stat-label">💡 Licht an</div></div>
                <div class="stat-item"><div class="stat-value off" id="lampOffCount">0</div><div class="stat-label">🌙 Licht aus</div></div>
                <div class="training-progress"><div class="progress-bar"><div class="progress-fill" id="warmupProgress" style="width:0%"></div></div><div class="stat-label" id="warmupText">0%</div></div>
            </div>
        </div>
        
        <div class="signal-container">
            <div class="signal-title" id="signalTitle">Elektromagnetisches Feld der Pflanze</div>
            <canvas id="signalCanvas"></canvas>
        </div>
        
        <div class="bottom-area">
            <div class="language-zone">
                <div class="gesture-overlay" id="gestureOverlay">
                    <video id="gestureVideo" playsinline></video>
                    <canvas id="gestureCanvas"></canvas>
                </div>
                <div class="gesture-feedback" id="gestureFeedback"></div>
                <div class="swipe-hint" id="swipeHint">👋 Wische für Sprache</div>
                <div class="language-buttons">
                    <button class="lang-btn active" data-lang="de">DE</button>
                    <button class="lang-btn" data-lang="en">EN</button>
                    <button class="lang-btn" data-lang="fr">FR</button>
                    <button class="lang-btn" data-lang="it">IT</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="qr-panel">
        <h4>Plantangle</h4>
        <div id="qrBox"></div>
        <div class="qr-label" id="qrLabel"></div>
        <div class="qr-hint">Scan for sensor IP</div>
    </div>
    
    <div class="staff-controls">
        <button class="staff-btn active" id="btnTraining" onclick="setMode('training')">Train</button>
        <button class="staff-btn" id="btnExhibition" onclick="setMode('exhibition')">Exhibit</button>
        <button class="staff-btn" onclick="retrain()">Reset</button>
    </div>
    
    <script>
        const plantWs = "{{ plant_ws }}";
        const socket = io();
        let currentLang = 'de', currentMode = 'training';
        const languages = ['de', 'en', 'fr', 'it'];
        
        const T = {
            de: { 
                title: 'LICHTWAHRNEHMUNG', 
                subtitle: 'Spüre ich Licht? Probiere es aus.',
                lampPanel: 'Die Lampe:',
                plantPanel: 'Das spürt die Pflanze:',
                sensing: 'Ich spüre...',
                btnOn: 'Licht einschalten', btnOff: 'Licht ausschalten',
                signalTitle: 'Elektromagnetisches Feld der Pflanze',
                swipeHint: '👋 Wische für Sprache',
                lampOn: 'An', lampOff: 'Aus',
                scale: ['Dunkel<br>sicher', 'Dunkel<br>wahrscheinlich', 'Licht<br>wahrscheinlich', 'Licht<br>sicher'],
                plant: { lamp_on: 'Licht!', lamp_off: 'Dunkelheit' }
            },
            en: { 
                title: 'LIGHT PERCEPTION', 
                subtitle: 'Can I sense light? Try it out.',
                lampPanel: 'The Lamp:',
                plantPanel: 'The plant senses:',
                sensing: 'I sense...',
                btnOn: 'Turn light on', btnOff: 'Turn light off',
                signalTitle: 'Electromagnetic field of the plant',
                swipeHint: '👋 Swipe for language',
                lampOn: 'On', lampOff: 'Off',
                scale: ['Dark<br>certain', 'Dark<br>probable', 'Light<br>probable', 'Light<br>certain'],
                plant: { lamp_on: 'Light!', lamp_off: 'Darkness' }
            },
            fr: { 
                title: 'PERCEPTION DE LA LUMIÈRE', 
                subtitle: "Est-ce que je perçois la lumière? Essayez.",
                lampPanel: 'La Lampe:',
                plantPanel: 'La plante ressent:',
                sensing: 'Je ressens...',
                btnOn: 'Allumer', btnOff: 'Éteindre',
                signalTitle: 'Champ électromagnétique de la plante',
                swipeHint: '👋 Glissez pour langue',
                lampOn: 'Allumée', lampOff: 'Éteinte',
                scale: ['Sombre<br>certain', 'Sombre<br>probable', 'Lumière<br>probable', 'Lumière<br>certaine'],
                plant: { lamp_on: 'Lumière!', lamp_off: 'Obscurité' }
            },
            it: { 
                title: 'PERCEZIONE DELLA LUCE', 
                subtitle: 'Percepisco la luce? Provaci.',
                lampPanel: 'La Lampada:',
                plantPanel: 'La pianta sente:',
                sensing: 'Sento...',
                btnOn: 'Accendi luce', btnOff: 'Spegni luce',
                signalTitle: 'Campo elettromagnetico della pianta',
                swipeHint: '👋 Scorri per lingua',
                lampOn: 'Accesa', lampOff: 'Spenta',
                scale: ['Buio<br>certo', 'Buio<br>probabile', 'Luce<br>probabile', 'Luce<br>certa'],
                plant: { lamp_on: 'Luce!', lamp_off: 'Oscurità' }
            }
        };
        
        const t = () => T[currentLang] || T.en;
        
        function updateLang(lang) {
            currentLang = lang;
            document.querySelectorAll('.lang-btn').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
            const tr = t();
            document.getElementById('exhibitTitle').textContent = tr.title;
            document.getElementById('exhibitSubtitle').textContent = tr.subtitle;
            document.getElementById('lampPanelTitle').textContent = tr.lampPanel;
            document.getElementById('plantPanelTitle').textContent = tr.plantPanel;
            document.getElementById('plantSensing').textContent = tr.sensing;
            document.getElementById('btnLampOn').textContent = tr.btnOn;
            document.getElementById('btnLampOff').textContent = tr.btnOff;
            document.getElementById('signalTitle').textContent = tr.signalTitle;
            document.getElementById('swipeHint').textContent = tr.swipeHint;
            document.getElementById('scaleLabel1').innerHTML = tr.scale[0];
            document.getElementById('scaleLabel2').innerHTML = tr.scale[1];
            document.getElementById('scaleLabel3').innerHTML = tr.scale[2];
            document.getElementById('scaleLabel4').innerHTML = tr.scale[3];
        }
        
        function cycleLang(dir) {
            const idx = languages.indexOf(currentLang);
            updateLang(languages[(idx + dir + 4) % 4]);
            showFeedback('🌐 ' + currentLang.toUpperCase());
        }
        
        document.querySelectorAll('.lang-btn').forEach(btn => btn.addEventListener('click', () => updateLang(btn.dataset.lang)));
        document.addEventListener('keydown', e => {
            if (e.key >= '1' && e.key <= '4') updateLang(languages[e.key - 1]);
            if (e.key === 'ArrowLeft') cycleLang(-1);
            if (e.key === 'ArrowRight') cycleLang(1);
        });
        
        const gestureVideo = document.getElementById('gestureVideo');
        const gestureCanvas = document.getElementById('gestureCanvas');
        const gestureCtx = gestureCanvas.getContext('2d');
        const gestureOverlay = document.getElementById('gestureOverlay');
        
        let handHistory = { left: [], right: [] }, langCooldown = false, animPhase = 0;
        const swipeThreshold = 1.2, histLen = 8;
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
            gestureCtx.save(); gestureCtx.clearRect(0, 0, w, h);
            gestureCtx.translate(w, 0); gestureCtx.scale(-1, 1);
            gestureCtx.fillStyle = '#0a0f14'; gestureCtx.fillRect(0, 0, w, h);
            let detected = false;
            if (results.poseLandmarks) {
                const lm = results.poseLandmarks;
                const vis = (lm[11].visibility + lm[12].visibility) / 2;
                if (vis > 0.5) { detected = true; drawAvatar(lm, w, h); detectGestures(lm); }
            }
            gestureOverlay.classList.toggle('detected', detected);
            gestureCtx.restore();
        }
        
        function drawAvatar(lm, w, h) {
            const [nose,lSh,rSh,lEl,rEl,lWr,rWr,lHip,rHip,lKn,rKn,lAn,rAn] = [0,11,12,13,14,15,16,23,24,25,26,27,28].map(i => lm[i]);
            const shC = { x:(lSh.x+rSh.x)/2*w, y:(lSh.y+rSh.y)/2*h };
            const hiC = { x:(lHip.x+rHip.x)/2*w, y:(lHip.y+rHip.y)/2*h };
            gestureCtx.strokeStyle = colors.stem; gestureCtx.lineCap = 'round'; gestureCtx.lineWidth = 8;
            [[lAn,lKn,lHip],[rAn,rKn,rHip]].forEach(([a,k,hp]) => { gestureCtx.beginPath(); gestureCtx.moveTo(a.x*w,a.y*h); gestureCtx.quadraticCurveTo(k.x*w,k.y*h,hp.x*w,hp.y*h); gestureCtx.stroke(); });
            const mx = (hiC.x+shC.x)/2+Math.sin(animPhase)*2, my = (hiC.y+shC.y)/2;
            gestureCtx.lineWidth = 10; gestureCtx.beginPath(); gestureCtx.moveTo(hiC.x,hiC.y); gestureCtx.quadraticCurveTo(mx,my,shC.x,shC.y); gestureCtx.stroke();
            drawLeaf(mx-8,my-10,-0.8,14); drawLeaf(mx+8,my+6,0.8,12);
            gestureCtx.lineWidth = 6;
            [[lSh,lEl,lWr,true],[rSh,rEl,rWr,false]].forEach(([sh,el,wr,isL]) => { gestureCtx.beginPath(); gestureCtx.moveTo(sh.x*w,sh.y*h); gestureCtx.lineTo(el.x*w,el.y*h); gestureCtx.lineTo(wr.x*w,wr.y*h); gestureCtx.stroke(); drawLeaf(el.x*w,el.y*h,isL?-0.6:0.6,12); for(let i=-1;i<=1;i++) drawLeaf(wr.x*w,wr.y*h,(isL?Math.PI:0)+i*0.4,10); });
            const hx=nose.x*w, hy=nose.y*h, r=24;
            gestureCtx.lineWidth = 7; gestureCtx.beginPath(); gestureCtx.moveTo(shC.x,shC.y); gestureCtx.lineTo(hx,hy+r*0.3); gestureCtx.stroke();
            for(let i=0;i<8;i++){gestureCtx.save();gestureCtx.translate(hx,hy);gestureCtx.rotate((i/8)*Math.PI*2+animPhase*0.04);gestureCtx.fillStyle=colors.flower;gestureCtx.beginPath();gestureCtx.ellipse(0,-r*0.4,r*0.28,r*0.45,0,0,Math.PI*2);gestureCtx.fill();gestureCtx.restore();}
            gestureCtx.fillStyle=colors.center;gestureCtx.beginPath();gestureCtx.arc(hx,hy,r*0.28,0,Math.PI*2);gestureCtx.fill();
        }
        
        function drawLeaf(x,y,ang,sz) {
            gestureCtx.save();gestureCtx.translate(x,y);gestureCtx.rotate(ang+Math.sin(animPhase+x*0.02)*0.05);
            gestureCtx.fillStyle=colors.leaf;gestureCtx.beginPath();gestureCtx.moveTo(0,0);
            gestureCtx.quadraticCurveTo(sz*0.5,-sz*0.26,sz,0);gestureCtx.quadraticCurveTo(sz*0.5,sz*0.26,0,0);
            gestureCtx.fill();gestureCtx.restore();
        }
        
        function detectGestures(lm) {
            const now=Date.now(),lWr=lm[15],rWr=lm[16];
            handHistory.left.push({x:lWr.x,y:lWr.y,t:now}); handHistory.right.push({x:rWr.x,y:rWr.y,t:now});
            if(handHistory.left.length>histLen) handHistory.left.shift();
            if(handHistory.right.length>histLen) handHistory.right.shift();
            if(handHistory.left.length<3) return;
            const lV=calcVel(handHistory.left),rV=calcVel(handHistory.right);
            const lSwR=lV.x<-swipeThreshold&&Math.abs(lV.y)<0.5;
            const rSwR=rV.x<-swipeThreshold&&Math.abs(rV.y)<0.5;
            const rStill=Math.abs(rV.x)<0.4,lStill=Math.abs(lV.x)<0.4;
            if(!langCooldown){
                if((lSwR&&rStill)||(rSwR&&lStill)){cycleLang(1);langCooldown=true;handHistory={left:[],right:[]};setTimeout(()=>langCooldown=false,1000);}
            }
        }
        
        function calcVel(h) {
            if(h.length<2) return {x:0,y:0};
            let vx=0,vy=0,tw=0;
            for(let i=1;i<h.length;i++){const dt=(h[i].t-h[i-1].t)/1000;if(dt>0&&dt<0.5){vx+=((h[i].x-h[i-1].x)/dt)*i;vy+=((h[i].y-h[i-1].y)/dt)*i;tw+=i;}}
            return tw?{x:vx/tw,y:vy/tw}:{x:0,y:0};
        }
        
        function showFeedback(txt) {
            const el=document.getElementById('gestureFeedback');
            el.textContent=txt;el.classList.remove('active');void el.offsetWidth;el.classList.add('active');
            setTimeout(()=>el.classList.remove('active'),500);
        }
        
        const sigCanvas=document.getElementById('signalCanvas'),sigCtx=sigCanvas.getContext('2d');
        function resizeSig(){
            const c=sigCanvas.parentElement;
            const w=c.clientWidth||c.offsetWidth||window.innerWidth;
            const h=(c.clientHeight||c.offsetHeight||80)-12;
            if(w>0&&h>0){sigCanvas.width=w;sigCanvas.height=h;}
        }
        function drawSignal(data,isLampOn) {
            resizeSig();const w=sigCanvas.width,h=sigCanvas.height;sigCtx.clearRect(0,0,w,h);
            if(data.length<2) return;
            const min=Math.min(...data),max=Math.max(...data),range=max-min||1,pad=range*0.1;
            sigCtx.beginPath();sigCtx.strokeStyle=isLampOn?'#fbbf24':'#1cffa8';sigCtx.lineWidth=1.5;
            for(let i=0;i<data.length;i++){const x=(i/data.length)*w,y=h-((data[i]-min+pad)/(range+pad*2))*h;i?sigCtx.lineTo(x,y):sigCtx.moveTo(x,y);}
            sigCtx.stroke();
        }
        
        const plantVideo=document.getElementById('plantVideo'); let vidState='dark';
        function switchVideo(s){if(vidState!==s){plantVideo.src=`/videos/${s}.mp4`;plantVideo.load();plantVideo.play().catch(()=>{});vidState=s;}}
        
        function lampOn(){fetch('/set_lamp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:true})}).then(r=>r.json()).then(d=>{if(d.status==='ok')showFeedback('💡');});}
        function lampOff(){fetch('/set_lamp',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:false})}).then(r=>r.json()).then(d=>{if(d.status==='ok')showFeedback('🌙');});}
        
        function updateMode(mode) {
            currentMode=mode;
            document.getElementById('btnTraining').classList.toggle('active',mode==='training');
            document.getElementById('btnExhibition').classList.toggle('active',mode==='exhibition');
            document.getElementById('trainingStats').classList.toggle('visible',mode==='training');
        }
        function setMode(mode){fetch('/toggle_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})}).then(r=>r.json()).then(d=>updateMode(d.mode));}
        function retrain(){fetch('/retrain',{method:'POST'});}
        
        socket.on('update', d => {
            document.getElementById('plantDot').classList.toggle('connected',d.connected);
            document.getElementById('plugDot').classList.toggle('connected',d.plug_connected);
            if(d.mode!==currentMode) updateMode(d.mode);
            const lampIcon=document.getElementById('lampIcon');
            const lampStatus=document.getElementById('lampStatus');
            const tr=t();
            lampIcon.classList.toggle('on',d.lamp_on);
            lampStatus.classList.toggle('on',d.lamp_on);
            lampStatus.textContent=d.lamp_on?tr.lampOn:tr.lampOff;
            document.getElementById('btnLampOn').disabled=d.lamp_on;
            document.getElementById('btnLampOff').disabled=!d.lamp_on;
            const vc=document.getElementById('videoContainer');
            const pr=document.getElementById('plantResponse');
            const sm=document.getElementById('scaleMarker');
            vc.classList.toggle('lamp-on',d.prediction==='lamp_on');
            vc.classList.toggle('collecting',d.collecting);
            if(d.prediction==='lamp_on'){switchVideo('light');pr.textContent=tr.plant.lamp_on;pr.className='plant-response lamp-on';}
            else{switchVideo('dark');pr.textContent=tr.plant.lamp_off;pr.className='plant-response lamp-off';}
            const scalePos=d.prediction==='lamp_on'?50+(d.confidence*50):50-(d.confidence*50);
            sm.style.left=scalePos+'%';
            document.getElementById('lampOnCount').textContent=d.lamp_on_count||0;
            document.getElementById('lampOffCount').textContent=d.lamp_off_count||0;
            document.getElementById('warmupProgress').style.width=(d.warmup_progress*100)+'%';
            document.getElementById('warmupText').textContent=Math.round(d.warmup_progress*100)+'%';
            if(d.waveform?.length>0) drawSignal(d.waveform,d.prediction==='lamp_on');
        });
        
        if(window.QRCode){
            try {
                new QRCode(document.getElementById('qrBox'),{text:plantWs,width:80,height:80,colorDark:"#1cffa8",colorLight:"#0a0f14",correctLevel:QRCode.CorrectLevel.M});
                document.getElementById('qrLabel').textContent=plantWs;
            } catch(e) { console.warn('QR error:', e); document.getElementById('qrLabel').textContent=plantWs; }
        } else {
            console.warn('QRCode library not loaded');
            document.getElementById('qrLabel').textContent=plantWs;
        }
        window.addEventListener('resize',resizeSig);
        setTimeout(()=>{ resizeSig(); }, 100);
        setTimeout(()=>{ resizeSig(); }, 500);
        initGesture();
        document.addEventListener('click',()=>plantVideo.play().catch(()=>{}),{once:true});
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

@app.route('/fonts/debug')
def debug_fonts():
    result = {'FONTS_DIR': str(FONTS_DIR), 'exists': FONTS_DIR.exists(), 'files': {}}
    if FONTS_DIR.exists():
        for p in sorted(FONTS_DIR.rglob('*')):
            if p.is_file():
                result['files'][str(p.relative_to(FONTS_DIR))] = p.stat().st_size
    return jsonify(result)

@app.route('/fonts/<path:filename>')
def serve_font(filename):
    fname_lower = filename.lower()
    search_dirs = [
        FONTS_DIR / "Satoshi Fonts", FONTS_DIR / "TownandCountryJNL", FONTS_DIR,
        Path("./Fonts/Satoshi Fonts"), Path("./Fonts/TownandCountryJNL"), Path("./Fonts"), Path("./fonts"),
    ]
    for d in search_dirs:
        if not d.exists(): continue
        exact = d / filename
        if exact.exists(): return send_from_directory(d, filename)
        for f in d.iterdir():
            if f.is_file() and f.name.lower() == fname_lower:
                return send_from_directory(d, f.name)
            if f.is_file() and f.suffix.lower() in ['.otf', '.ttf', '.woff', '.woff2']:
                if fname_lower.replace('-', '').replace(' ', '').split('.')[0] in f.name.lower().replace('-', '').replace(' ', ''):
                    return send_from_directory(d, f.name)
    print(f"⚠️  Font not found: {filename}")
    return "Font not found", 404

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.get_json(silent=True) or {}
    requested_mode = data.get('mode')
    if requested_mode == 'exhibition' and state.model.is_trained:
        state.mode = 'exhibition'
    elif requested_mode == 'training':
        state.mode = 'training'
        state.next_toggle_time = time.time() + 5
    return jsonify({'mode': state.mode})

@app.route('/toggle_lamp', methods=['POST'])
def toggle_lamp():
    global shelly_plug
    if not shelly_plug or not shelly_plug.connected:
        return jsonify({'status': 'error', 'error': 'Plug not connected'})
    
    new_state = not state.lamp_on
    success = shelly_plug.turn_on() if new_state else shelly_plug.turn_off()
    
    if success:
        state.lamp_on = new_state
        state.lamp_toggle_time = time.time()
        state.training_phase = 'waiting'
        return jsonify({'status': 'ok', 'lamp_on': state.lamp_on})
    return jsonify({'status': 'error'})

@app.route('/set_lamp', methods=['POST'])
def set_lamp():
    global shelly_plug
    if not shelly_plug or not shelly_plug.connected:
        return jsonify({'status': 'error', 'error': 'Plug not connected'})
    data = request.get_json(silent=True) or {}
    new_state = data.get('state', False)
    if new_state == state.lamp_on:
        return jsonify({'status': 'ok', 'lamp_on': state.lamp_on})
    success = shelly_plug.turn_on() if new_state else shelly_plug.turn_off()
    if success:
        state.lamp_on = new_state
        state.lamp_toggle_time = time.time()
        state.training_phase = 'waiting'
        return jsonify({'status': 'ok', 'lamp_on': state.lamp_on})
    return jsonify({'status': 'error'})

@app.route('/retrain', methods=['POST'])
def retrain():
    state.model.force_retrain()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 PHAENOMENA - LICHTWAHRNEHMUNG")
    print("   Unified Exhibition Interface with Gesture Controls")
    print("="*60)
    print(f"🌐 http://localhost:5002")
    print("="*60 + "\n")
    
    shelly_plug = ShellyPlug(SHELLY_IP)
    if shelly_plug.connected:
        state.plug_connected = True
        state.lamp_on = shelly_plug.last_status
        print("✅ Shelly plug connected")
    else:
        print("⚠️ Shelly plug not connected")
        shelly_plug = None
    
    loaded, age = state.model.load_latest()
    if loaded and age < MODEL_MAX_AGE_MINUTES:
        state.mode = 'exhibition'
        print("✨ Model fresh - exhibition mode")
    else:
        state.next_toggle_time = time.time() + 5
        print("🔧 Training mode")
    
    threading.Thread(target=run_esp32_listener, daemon=True).start()
    threading.Thread(target=training_lamp_controller, daemon=True).start()
    threading.Thread(target=broadcast_updates, daemon=True).start()
    
    socketio.run(app, host='0.0.0.0', port=5002, debug=False, allow_unsafe_werkzeug=True)
