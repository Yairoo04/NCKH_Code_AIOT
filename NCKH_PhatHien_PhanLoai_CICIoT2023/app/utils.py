import logging
from queue import Queue, Empty
import argparse
import sys
import psutil
import socket
import os

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'all_log.log'))
    ]
)
logger = logging.getLogger('DDoSDetector')

attack_logger = logging.getLogger('DDoSDetector.Attack')
attack_logger.setLevel(logging.INFO)
attack_handler = logging.FileHandler(os.path.join(log_dir, 'attack.log'))
attack_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
attack_logger.addHandler(attack_handler)

no_attack_logger = logging.getLogger('DDoSDetector.NoAttack')
no_attack_logger.setLevel(logging.INFO)
no_attack_handler = logging.FileHandler(os.path.join(log_dir, 'no_attack.log'))
no_attack_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
no_attack_logger.addHandler(no_attack_handler)

update_logger = logging.getLogger('DDoSDetector.Update')
update_logger.setLevel(logging.INFO)
update_handler = logging.FileHandler(os.path.join(log_dir, 'update_log.log'))
update_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
update_logger.addHandler(update_handler)

detector_queue = Queue(maxsize=1000)

def detector_updater(queue, detector):
    while detector.running:
        try:
            features, is_attack, priority, probability = queue.get(timeout=0.1)
            if not isinstance(features, dict):
                logger.error(f"Invalid features format: {features}")
                continue
            detector.add_sample(features, is_attack, probability)
            queue.task_done()
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Error in detector_updater: {e}", exc_info=True)

def signal_handler(sig, frame):
    logger.info("Shutting down application...")
    sys.exit(0)

def get_default_interface():
    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                return name
    return "eth0"

def parse_args():
    parser = argparse.ArgumentParser(description='DDoS Detection System')
    parser.add_argument('-i', '--interface', help='Network interface to capture packets', default=get_default_interface())
    parser.add_argument('-w', '--window', type=float, help='Time window size for traffic analysis (seconds)', default=1.0)
    parser.add_argument('-r', '--retention', type=int, help='Data retention time (minutes)', default=10)
    parser.add_argument('-u', '--update', type=int, help='Dashboard update interval (seconds)', default=1)
    parser.add_argument('-m', '--model', help='Path to ML model file', default="models/aggregated_model.pt")
    parser.add_argument('--scaler', help='Path to scaler file', default="models/scaler_server.pkl")
    parser.add_argument('--encoder', help='Path to label encoder file', default="models/label_encoder_server.pkl")
    parser.add_argument('-p', '--port', type=int, help='Web server port', default=5000)
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        attack_logger.setLevel(logging.DEBUG)
        no_attack_logger.setLevel(logging.DEBUG)
        update_logger.setLevel(logging.DEBUG)

    return args