from flask import Flask, render_template, jsonify, request
from threading import Thread
import signal
import sys
from config import Config
from detector import DDoSDetector
from analyzer import TrafficAnalyzer, run_traffic_analyzer
from utils import detector_queue, detector_updater, signal_handler, parse_args, logger
from dashboard import setup_dashboard

flask_app = Flask(__name__)
config = Config()
detector = DDoSDetector()

dash_app = setup_dashboard(flask_app, detector, config)

@flask_app.route('/')
def home():
    """Trang chủ Flask"""
    return render_template('index.html')

@flask_app.route('/api/status')
def get_status():
    """API trả về trạng thái hệ thống"""
    try:
        return jsonify({'status': detector.current_status})
    except Exception as e:
        logger.error(f"Error in get_status: {e}", exc_info=True)
        return jsonify({'status': 'Error'})

@flask_app.route('/api/metrics')
def get_metrics():
    """API trả về các metrics hiện tại"""
    try:
        recent_data = detector.get_recent_data(minutes=1)
        if recent_data.empty:
            return jsonify({
                'flow_bytes_s': 0,
                'flow_packets_s': 0,
                'unique_sources': 0,
                'is_attack': False
            })
            
        return jsonify({
            'flow_bytes_s': float(recent_data['flow bytes/s'].mean()),
            'flow_packets_s': float(recent_data['flow packets/s'].mean()),
            'unique_sources': int(recent_data['source ip'].nunique()),
            'is_attack': detector.current_status == "Under Attack"
        })
    except Exception as e:
        logger.error(f"Error in get_metrics: {e}", exc_info=True)
        return jsonify({
            'flow bytes/s': 0,
            'flow packets/s': 0,
            'unique_sources': 0,
            'is_attack': False
        })

@flask_app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """API để lấy và cập nhật cấu hình"""
    if request.method == 'GET':
        return jsonify({
            'interface': config.interface,
            'window_size': config.window_size,
            'data_retention_minutes': config.data_retention_minutes,
            'dashboard_update_interval': config.dashboard_update_interval
        })
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if 'window_size' in data:
                config.window_size = float(data['window_size'])
            if 'data_retention_minutes' in data:
                config.data_retention_minutes = int(data['data_retention_minutes'])
            if 'dashboard_update_interval' in data:
                config.dashboard_update_interval = int(data['dashboard_update_interval'])
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error updating config: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)})
        
@flask_app.route('/api/unblock', methods=['POST'])
def unblock_ip():
    """API để bỏ chặn IP"""
    try:
        data = request.get_json()
        ip = data.get('ip')
        if not ip:
            return jsonify({'status': 'error', 'message': 'IP address required'}), 400
        if ip in detector.blocked_ips:
            detector._unblock_ip(ip)
            return jsonify({'status': 'success', 'message': f'Unblocked IP {ip}'})
        return jsonify({'status': 'error', 'message': f'IP {ip} not blocked'}), 404
    except Exception as e:
        logger.error(f"Error in unblock_ip: {e}", exc_info=True)
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

    logger.info(f"Starting web server on {config.host}:{config.port}")
    flask_app.run(
        debug=config.debug,
        host=config.host,
        port=config.port,
        threaded=True
    )