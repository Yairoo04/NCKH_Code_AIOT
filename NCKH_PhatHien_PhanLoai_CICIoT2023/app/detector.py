import pandas as pd
import time
from threading import Lock
import logging
import socket
import threading
import re

logger = logging.getLogger('DDoSDetector')

class DDoSDetector:
    def __init__(self):
        self.traffic_data = pd.DataFrame({
            'timestamp': [], 'flow id': [], 'source ip': [], 'destination ip': [],
            'Header_Length': [], 'Protocol Type': [], 'Time_To_Live': [], 'Rate': [],
            'fin_flag_number': [], 'syn_flag_number': [], 'rst_flag_number': [],
            'psh_flag_number': [], 'ack_flag_number': [], 'HTTP': [], 'HTTPS': [],
            'TCP': [], 'UDP': [], 'ICMP': [], 'Tot sum': [], 'Min': [], 'Max': [],
            'AVG': [], 'Std': [], 'Tot size': [], 'IAT': [], 'Number': [],
            'is_attack': [], 'attack_type': []
        })
        self.alerts = []
        self.current_status = "Normal"
        self.blocked_ips = set()
        self.attack_stats = {'total_attacks': 0, 'blocked_ips': 0, 'last_attack': None}
        self.data_lock = Lock()
        self.temp_data = []
        self.running = True
        self.local_ip = socket.gethostbyname(socket.gethostname())
        logger.info(f"Local IP detected: {self.local_ip}")
        self.last_attack_time = None
        self.attack_grace_period = 10
        
    def add_sample(self, flow_data, is_attack):
        try:
            required_fields = ['source ip', 'destination ip', 'Rate', 'attack_type']
            for field in required_fields:
                if field not in flow_data:
                    logger.error(f"Missing field {field} in flow_data: {flow_data}")
                    return

            flow_data['timestamp'] = time.time()
            flow_data['is_attack'] = is_attack

            with self.data_lock:
                self.temp_data.append(flow_data)
                if len(self.temp_data) >= 100 or is_attack:
                    self._update_traffic_data()

            if is_attack:
                self._handle_attack(flow_data)
            else:
                self._check_normal_status()
        except Exception as e:
            logger.error(f"Error in add_sample: {e}", exc_info=True) 

    def _update_traffic_data(self):
        try:
            if not self.temp_data:
                return
                
            new_data = pd.DataFrame(self.temp_data)
            for col in self.traffic_data.columns:
                if col not in new_data.columns:
                    new_data[col] = None
                    
            self.traffic_data = pd.concat([self.traffic_data, new_data], ignore_index=True)
            self.temp_data = []
            
            cutoff = time.time() - (10 * 60)
            self.traffic_data = self.traffic_data[self.traffic_data['timestamp'] > cutoff]
            logger.debug(f"Updated traffic_data, Rows now: {len(self.traffic_data)}")
        except Exception as e:
            logger.error(f"Error in _update_traffic_data: {e}", exc_info=True)

    def _handle_attack(self, flow_data):
        with self.data_lock:
            source_ip = flow_data['source ip']
            if source_ip == self.local_ip:
                logger.info(f"Ignoring attack from local IP: {source_ip}")
                return

            attack_type = flow_data['attack_type']
            self.alerts.append({
                'time': time.strftime('%H:%M:%S'),
                'source ip': source_ip,
                'message': f'DDoS Attack Detected! Type: {attack_type}, Source IP: {source_ip}, Rate: {flow_data["Rate"]:.2f} packets/s',
                'mitigation': f'Suggested: Block traffic from {source_ip}'
            })
            
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
                
            self.current_status = f"Under Attack ({attack_type})"
            self.last_attack_time = time.time()
            self.attack_stats['total_attacks'] += 1
            self.attack_stats['last_attack'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            if (source_ip not in self.blocked_ips and 
                flow_data['Rate'] > 1000):
                self.blocked_ips.add(source_ip)
                self.attack_stats['blocked_ips'] = len(self.blocked_ips)
                self._block_ip(source_ip)
                self.alerts[-1]['mitigation'] = f'Blocked traffic from {source_ip}'
                logger.warning(f"Blocked IP: {source_ip}")
                
            logger.warning(f"Attack detected: Type={attack_type}, Src={source_ip}, Rate={flow_data['Rate']:.2f}, "
                          f"Blocked={source_ip in self.blocked_ips}")

    def _block_ip(self, ip):
        import os
        try:
            cmd = f"iptables -A INPUT -s {ip} -p udp --dport 5002 -j DROP"
            os.system(cmd)
            logger.info(f"Successfully blocked IP {ip} using iptables")
        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {e}", exc_info=True)

    def _unblock_ip(self, ip):
        import os
        try:
            cmd = f"iptables -D INPUT -s {ip} -p udp --dport 5002 -j DROP"
            os.system(cmd)
            self.blocked_ips.discard(ip)
            self.attack_stats['blocked_ips'] = len(self.blocked_ips)
            logger.info(f"Successfully unblocked IP {ip}")
        except Exception as e:
            logger.error(f"Failed to unblock IP {ip}: {e}", exc_info=True)

    def _check_normal_status(self):
        with self.data_lock:
            if self.last_attack_time is not None:
                time_since_last_attack = time.time() - self.last_attack_time
                if time_since_last_attack > self.attack_grace_period:
                    self.current_status = "Normal"
                    logger.info("Status changed to Normal")
            elif len(self.alerts) == 0:
                self.current_status = "Normal"

    def get_recent_data(self, minutes=5):
        try:
            with self.data_lock:
                if self.temp_data:
                    self._update_traffic_data()
                    
                cutoff = time.time() - (minutes * 60)
                recent = self.traffic_data[self.traffic_data['timestamp'] > cutoff]
                
                if 'is_attack' not in recent.columns or recent.empty:
                    return pd.DataFrame(columns=self.traffic_data.columns)
                    
                return recent.copy()
        except Exception as e:
            logger.error(f"Error in get_recent_data: {e}", exc_info=True)
            return pd.DataFrame()

    def get_alerts(self, count=20):
        with self.data_lock:
            return self.alerts[-count:]

    def get_ip_attack_counts(self):
        with self.data_lock:
            ip_counts = {}
            for alert in self.alerts:
                ip = self.extract_ip_from_message(alert['message'])
                if ip:
                    ip_counts[ip] = ip_counts.get(ip, 0) + 1
            return ip_counts

    def extract_ip_from_message(self, message):
        ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        match = re.search(ip_pattern, message)
        return match.group(0) if match else None

    def get_attack_stats(self):
        with self.data_lock:
            stats = self.attack_stats.copy()
            stats['blocked_ips_list'] = list(self.blocked_ips)
            return stats

    def get_top_ips(self, count=5):
        try:
            recent_data = self.get_recent_data()
            if recent_data.empty:
                return []
                
            ip_counts = recent_data['source ip'].value_counts().head(count)
            return [{'ip': ip, 'count': count} for ip, count in ip_counts.items()]
        except Exception as e:
            logger.error(f"Error in get_top_ips: {e}", exc_info=True)
            return []

    def get_feature_importance(self):
        try:
            model = None
            for thread in threading.enumerate():
                if hasattr(thread, '_target') and thread._target is not None and thread._target.__name__ == 'run_traffic_analyzer':
                    if hasattr(thread, '_args') and len(thread._args) > 0:
                        analyzer = thread._args[0]
                        if hasattr(analyzer, 'model'):
                            model = analyzer.model
                            break
            
            if model and hasattr(model, 'feature_importances_'):
                features = [
                    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
                    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
                    'ack_flag_number', 'HTTP', 'HTTPS', 'TCP', 'UDP', 'ICMP',
                    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number'
                ]
                importances = model.feature_importances_
                
                min_len = min(len(features), len(importances))
                features = features[:min_len]
                importances = importances[:min_len]
                
                importance_dict = dict(zip(features, importances))
                return features, importance_dict
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}", exc_info=True)
            return None
    
    def shutdown(self):
        self.running = False