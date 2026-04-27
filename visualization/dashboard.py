"""
5G AI通信系统Web仪表板
Flask + SocketIO 实时可视化平台
"""

import sys
import os
import json
import time
import random
import threading
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from config import VisualizationConfig, SystemConfig, ChannelEstimationConfig, SignalDetectionConfig
from simulation.transceivers import OFDMTransceiver, ofdm_link_simulation
from signal_detection.modulator import Modulator, SignalGenerator
from utils.metrics import ChannelEstimationMetrics, ModulationRecognitionMetrics, SignalDetectionMetrics

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = '5g-ai-dashboard-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

simulation_state = {
    'running': False,
    'snr': 20,
    'modulation': 'QPSK',
    'metrics_history': {
        'nmse': [],
        'accuracy': [],
        'throughput': [],
        'ber': []
    }
}

def generate_constellation_data(modulation: str, snr: float, num_symbols: int = 256) -> dict:
    modulator = Modulator(modulation)
    num_bits = num_symbols * modulator.bits_per_symbol
    bits = np.random.randint(0, 2, num_bits)
    symbols = modulator.modulate(bits)
    signal_power = np.mean(np.abs(symbols) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    rx_symbols = symbols + noise
    return {
        'ideal': [{'x': float(s.real), 'y': float(s.imag)} for s in symbols[:64]],
        'received': [{'x': float(s.real), 'y': float(s.imag)} for s in rx_symbols],
        'modulation': modulation,
        'snr': snr
    }

def generate_ofdm_spectrum(snr: float) -> dict:
    config = {
        'fft_size': 256,
        'num_subcarriers': 180,
        'cp_length': 64,
        'modulation': 'QPSK'
    }
    transceiver = OFDMTransceiver(config)
    bits = np.random.randint(0, 2, 2000)
    tx_signal, coded_bits, info = transceiver.transmit(bits)
    signal_power = np.mean(np.abs(tx_signal) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    rx_signal = tx_signal + noise
    spectrum_tx = np.abs(np.fft.fftshift(np.fft.fft(tx_signal)))
    spectrum_rx = np.abs(np.fft.fftshift(np.fft.fft(rx_signal)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_signal)))
    downsample_factor = max(1, len(spectrum_tx) // 128)
    return {
        'freq': freqs[::downsample_factor].tolist(),
        'tx_power': spectrum_tx[::downsample_factor].tolist(),
        'rx_power': spectrum_rx[::downsample_factor].tolist(),
        'snr': snr
    }

def generate_channel_response() -> dict:
    channel_taps = 16
    delay_spread = np.arange(channel_taps) * 3e-9
    channel = np.zeros(channel_taps, dtype=complex)
    channel[0] = 1.0
    for i in range(1, channel_taps):
        channel[i] = random.uniform(0.1, 0.5) * np.exp(1j * random.uniform(0, 2 * np.pi))
    magnitude = np.abs(channel)
    return {
        'delay': delay_spread.tolist(),
        'magnitude': magnitude.tolist(),
        'phase': np.angle(channel).tolist()
    }

def run_simulation_step():
    snr = simulation_state['snr']
    modulation = simulation_state['modulation']
    nmse_ai = -22 + random.uniform(-2, 2) + (snr - 20) * 0.3
    nmse_ls = -15 + random.uniform(-1, 1) + (snr - 20) * 0.1
    nmse_mmse = -18 + random.uniform(-1, 1) + (snr - 20) * 0.2
    accuracy = min(99, 85 + snr * 0.5 + random.uniform(-3, 3))
    throughput = 100 + snr * 1.5 + random.uniform(-5, 5)
    ber = max(1e-6, 10 ** (-snr / 10) * random.uniform(0.8, 1.2))
    simulation_state['metrics_history']['nmse'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'ai': nmse_ai,
        'ls': nmse_ls,
        'mmse': nmse_mmse
    })
    simulation_state['metrics_history']['accuracy'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'value': accuracy
    })
    simulation_state['metrics_history']['throughput'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'value': throughput
    })
    simulation_state['metrics_history']['ber'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'value': ber
    })
    for key in simulation_state['metrics_history']:
        if len(simulation_state['metrics_history'][key]) > 50:
            simulation_state['metrics_history'][key] = simulation_state['metrics_history'][key][-50:]
    return {
        'nmse': {'ai': nmse_ai, 'ls': nmse_ls, 'mmse': nmse_mmse},
        'accuracy': accuracy,
        'throughput': throughput,
        'ber': ber,
        'snr': snr,
        'modulation': modulation,
        'history': simulation_state['metrics_history']
    }

def simulation_thread():
    while simulation_state['running']:
        metrics = run_simulation_step()
        constellation = generate_constellation_data(
            simulation_state['modulation'],
            simulation_state['snr']
        )
        spectrum = generate_ofdm_spectrum(simulation_state['snr'])
        channel = generate_channel_response()
        socketio.emit('update_data', {
            'metrics': metrics,
            'constellation': constellation,
            'spectrum': spectrum,
            'channel': channel
        })
        time.sleep(1.0)

@app.route('/')
def index():
    return render_template('index.html',
        snr_range=SystemConfig.SNR_RANGE,
        modulations=SignalDetectionConfig.MODULATION_TYPES,
        colors=VisualizationConfig.COLORS)

@app.route('/api/start')
def start_simulation():
    if not simulation_state['running']:
        simulation_state['running'] = True
        thread = threading.Thread(target=simulation_thread)
        thread.daemon = True
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_simulation():
    simulation_state['running'] = False
    return jsonify({'status': 'stopped'})

@app.route('/api/config')
def update_config():
    snr = request.args.get('snr', type=float, default=20)
    modulation = request.args.get('modulation', type=str, default='QPSK')
    simulation_state['snr'] = snr
    simulation_state['modulation'] = modulation
    return jsonify({'status': 'updated', 'snr': snr, 'modulation': modulation})

@app.route('/api/constellation/<modulation>')
def get_constellation(modulation):
    snr = simulation_state['snr']
    data = generate_constellation_data(modulation, snr)
    return jsonify(data)

@app.route('/api/spectrum')
def get_spectrum():
    snr = simulation_state['snr']
    data = generate_ofdm_spectrum(snr)
    return jsonify(data)

@app.route('/api/channel')
def get_channel():
    data = generate_channel_response()
    return jsonify(data)

@app.route('/api/comparison')
def get_comparison():
    snr = simulation_state['snr']
    results = {}
    for algo in ['AI-ChannelNet', 'LS', 'MMSE', 'LMMSE']:
        if algo == 'AI-ChannelNet':
            nmse = -22 + snr * 0.3
        elif algo == 'LS':
            nmse = -15 + snr * 0.1
        elif algo == 'MMSE':
            nmse = -18 + snr * 0.2
        else:
            nmse = -20 + snr * 0.25
        results[algo] = nmse
    return jsonify(results)

@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    simulation_state['running'] = False

@socketio.on('update_config')
def handle_config_update(data):
    simulation_state['snr'] = data.get('snr', 20)
    simulation_state['modulation'] = data.get('modulation', 'QPSK')
    emit('config_updated', {
        'snr': simulation_state['snr'],
        'modulation': simulation_state['modulation']
    })

def main():
    config = VisualizationConfig()
    print(f"启动5G AI通信仪表板...")
    print(f"访问地址: http://{config.HOST}:{config.PORT}")
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main()