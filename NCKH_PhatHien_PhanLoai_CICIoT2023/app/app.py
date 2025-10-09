from flask import Flask, render_template, jsonify, request
from threading import Thread
import signal
import sys
from config import Config
from detector import DDoSDetector
from analyzer import TrafficAnalyzer, run_traffic_analyzer
from dashboard import setup_dashboard
from utils import detector_queue, detector_updater, signal_handler, parse_args, logger, update_logger

flask_app = Flask(__name__)
config = Config()
detector = DDoSDetector()

dash_app = setup_dashboard(flask_app, detector, config)

@flask_app.route('/')
def home():
    update_logger.info("Accessing home page")
    return render_template('index.html')

@flask_app.route('/api/status')
def get_status():
    try:
        update_logger.info("Fetching system status")
        return jsonify({'status': detector.current_status})
    except Exception as e:
        logger.error(f"Error in get_status: {e}", exc_info=True)
        update_logger.error(f"Error in get_status: {e}", exc_info=True)
        return jsonify({'status': 'Error'})

@flask_app.route('/api/metrics')
def get_metrics():
    try:
        update_logger.info("Fetching metrics")
        recent_data = detector.get_recent_data(minutes=1)
        if recent_data.empty:
            return jsonify({
                'flow_bytes_s': 0,
                'flow_packets_s': 0,
                'unique_sources': 0,
                'is_attack': False
            })
            
        return jsonify({
            'flow_bytes_s': float(recent_data['Tot size'].mean() if 'Tot size' in recent_data else 0),
            'flow_packets_s': float(recent_data['Rate'].mean() if 'Rate' in recent_data else 0),
            'unique_sources': int(recent_data['source ip'].nunique()),
            'is_attack': detector.current_status != "Normal"
        })
    except Exception as e:
        logger.error(f"Error in get_metrics: {e}", exc_info=True)
        update_logger.error(f"Error in get_metrics: {e}", exc_info=True)
        return jsonify({
            'flow_bytes_s': 0,
            'flow_packets_s': 0,
            'unique_sources': 0,
            'is_attack': False
        })

@flask_app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        update_logger.info("Fetching configuration")
        return jsonify({
            'interface': config.interface,
            'window_size': config.window_size,
            'data_retention_minutes': config.data_retention_minutes,
            'dashboard_update_interval': config.dashboard_update_interval
        })
    elif request.method == 'POST':
        try:
            data = request.get_json()
            update_logger.info(f"Updating configuration: {data}")
            if 'window_size' in data:
                config.window_size = float(data['window_size'])
            if 'data_retention_minutes' in data:
                config.data_retention_minutes = int(data['data_retention_minutes'])
            if 'dashboard_update_interval' in data:
                config.dashboard_update_interval = int(data['dashboard_update_interval'])
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error updating config: {e}", exc_info=True)
            update_logger.error(f"Error updating config: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)})
        
@flask_app.route('/api/unblock', methods=['POST'])
def unblock_ip():
    try:
        data = request.get_json()
        ip = data.get('ip')
        update_logger.info(f"Attempting to unblock IP: {ip}")
        if not ip:
            return jsonify({'status': 'error', 'message': 'IP address required'}), 400
        if ip in detector.blocked_ips:
            detector._unblock_ip(ip)
            update_logger.info(f"Unblocked IP: {ip}")
            return jsonify({'status': 'success', 'message': f'Unblocked IP {ip}'})
        return jsonify({'status': 'error', 'message': f'IP {ip} not blocked'}), 404
    except Exception as e:
        logger.error(f"Error in unblock_ip: {e}", exc_info=True)
        update_logger.error(f"Error in unblock_ip: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    updater_thread = Thread(
        target=detector_updater,
        args=(detector_queue, detector),
        name="DetectorUpdaterThread"
    )
    updater_thread.daemon = True
    updater_thread.start()
    logger.info("Detector updater thread started")
    update_logger.info("Detector updater thread started")

    analyzer = TrafficAnalyzer(
        interface=config.interface,
        window_size=config.window_size,
        detector=detector
    )
    capture_thread = Thread(
        target=run_traffic_analyzer,
        args=(analyzer,),
        name="TrafficCaptureThread"
    )
    capture_thread.daemon = True
    capture_thread.start()
    logger.info("Traffic capture thread started")
    update_logger.info("Traffic capture thread started")

    logger.info(f"Starting web server on {config.host}:{config.port}")
    update_logger.info(f"Starting web server on {config.host}:{config.port}")
    flask_app.run(
        debug=config.debug,
        host=config.host,
        port=config.port,
        threaded=True
    )