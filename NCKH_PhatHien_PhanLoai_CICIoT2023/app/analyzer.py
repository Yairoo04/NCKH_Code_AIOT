import time
from threading import Thread
from collections import defaultdict
import logging
import socket
from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

logger = logging.getLogger('DDoSDetector')

class TrafficAnalyzer:
    def __init__(self, interface, window_size=1, detector=None):
        self.interface = interface
        self.window_size = window_size
        self.flows = defaultdict(list)
        self.start_time = time.time()
        self.model = self.load_model()
        self.detector = detector
        self.running = True
        self.local_ip = socket.gethostbyname(socket.gethostname())
        logger.info(f"Local IP for filtering: {self.local_ip}")

    def load_model(self):
        try:
            model_path = os.path.join("models", "model_rf.pkl")
            scaler_path = os.path.join("models", "scaler.pkl")
            encoder_path = os.path.join("models", "label_encoder.pkl")
            
            for path in [model_path, scaler_path, encoder_path]:
                if not os.path.exists(path):
                    logger.warning(f"Không tìm thấy tệp: {path}")
                    return None
                if os.path.getsize(path) == 0:
                    logger.warning(f"Tệp {path} bị rỗng")
                    return None
            
            logger.info(f"Đang tải model từ {model_path}, scaler từ {scaler_path}, encoder từ {encoder_path}...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
            logger.info("Đã tải thành công model, scaler và label encoder")
            
            self.scaler = scaler
            self.label_encoder = label_encoder
            return model
        except Exception as e:
            logger.error(f"Lỗi khi tải model/scaler/encoder: {e}", exc_info=True)
            return None

    def packet_callback(self, packet):
        try:
            if not self.running or IP not in packet:
                return

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = 'TCP' if TCP in packet else 'UDP' if UDP in packet else 'ICMP' if ICMP in packet else 'Other'
            src_port = packet[TCP].sport if TCP in packet else packet[UDP].sport if UDP in packet else 0
            dst_port = packet[TCP].dport if TCP in packet else packet[UDP].dport if UDP in packet else 0
            flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
            
            logger.debug(f"Captured packet: {src_ip} -> {dst_ip}, Protocol: {protocol}")

            pkt_data = {
                'timestamp': time.time(),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'packet_length': len(packet),
                'ip_header_len': packet[IP].ihl * 4,
                'tcp_header_len': packet[TCP].dataofs * 4 if TCP in packet else 0,
                'init_win': packet[TCP].window if TCP in packet else 0,
                'is_http': (dst_port == 80 or src_port == 80 or dst_port == 443 or src_port == 443),
                'ttl': packet[IP].ttl if IP in packet else 0,
                'flags': packet[TCP].flags if TCP in packet else 0
            }
            
            self.flows[flow_id].append(pkt_data)

            if len(self.flows[flow_id]) >= 60 or (time.time() - self.start_time >= 1.0):
                self._process_flows_for_single_flow(flow_id)
                self.start_time = time.time()

        except Exception as e:
            logger.error(f"Error in packet_callback: {e}", exc_info=True)

    def _process_flows_for_single_flow(self, flow_id):
        if flow_id in self.flows:
            features_by_flow = self.extract_features_by_flow(flow_id)
            self.process_and_predict(features_by_flow)
            del self.flows[flow_id]

    def extract_features_by_flow(self, flow_id=None):
        features_by_flow = {}
        
        if flow_id:
            if flow_id in self.flows:
                target_flows = [(flow_id, self.flows[flow_id])]
            else:
                return features_by_flow
        else:
            target_flows = self.flows.items()
        
        for fid, packets in target_flows:
            total_packets = len(packets)
            if total_packets == 0:
                continue

            timestamps = sorted(pkt['timestamp'] for pkt in packets)
            time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else self.window_size
            time_span = max(time_span, 0.001)
            
            flow_packets_s = total_packets / time_span
            total_bytes = sum(pkt['packet_length'] for pkt in packets)
            flow_bytes_s = total_bytes / time_span

            src_ip = packets[0]['src_ip']
            dst_ip = packets[0]['dst_ip']
            
            header_length = sum(pkt['ip_header_len'] + pkt['tcp_header_len'] for pkt in packets)
            
            protocol = 'TCP' if any('tcp_header_len' in pkt and pkt['tcp_header_len'] > 0 for pkt in packets) else \
                       'UDP' if any('udp' in pkt for pkt in packets) else \
                       'ICMP' if any('icmp' in pkt for pkt in packets) else 'Other'
            
            fin_flag = 1 if any(pkt.get('flags', 0) & 0x01 for pkt in packets) else 0
            syn_flag = 1 if any(pkt.get('flags', 0) & 0x02 for pkt in packets) else 0
            rst_flag = 1 if any(pkt.get('flags', 0) & 0x04 for pkt in packets) else 0
            psh_flag = 1 if any(pkt.get('flags', 0) & 0x08 for pkt in packets) else 0
            ack_flag = 1 if any(pkt.get('flags', 0) & 0x10 for pkt in packets) else 0
            
            is_http = 1 if any(pkt['is_http'] for pkt in packets) else 0
            is_https = 1 if any(pkt.get('dst_port') == 443 or pkt.get('src_port') == 443 for pkt in packets) else 0
            
            is_tcp = 1 if protocol == 'TCP' else 0
            is_udp = 1 if protocol == 'UDP' else 0
            is_icmp = 1 if protocol == 'ICMP' else 0
            
            packet_sizes = [pkt['packet_length'] for pkt in packets]
            tot_size = sum(packet_sizes)
            min_size = min(packet_sizes) if packet_sizes else 0
            max_size = max(packet_sizes) if packet_sizes else 0
            avg_size = tot_size / total_packets if total_packets > 0 else 0
            tot_sum = tot_size
            
            iat = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
            
            ttl = max(pkt['ttl'] for pkt in packets) if any(pkt.get('ttl') for pkt in packets) else 0
            std = np.std(packet_sizes) if len(packet_sizes) > 1 else 0
            
            features_by_flow[fid] = {
                'flow id': fid,
                'source ip': src_ip,
                'destination ip': dst_ip,
                'Header_Length': header_length,
                'Protocol Type': protocol,
                'Time_To_Live': ttl,
                'Rate': flow_packets_s,
                'fin_flag_number': fin_flag,
                'syn_flag_number': syn_flag,
                'rst_flag_number': rst_flag,
                'psh_flag_number': psh_flag,
                'ack_flag_number': ack_flag,
                'HTTP': is_http,
                'HTTPS': is_https,
                'TCP': is_tcp,
                'UDP': is_udp,
                'ICMP': is_icmp,
                'Tot sum': tot_sum,
                'Min': min_size,
                'Max': max_size,
                'AVG': avg_size,
                'Std': std,
                'Tot size': tot_size,
                'IAT': iat,
                'Number': total_packets,
                'attack_type': None
            }
        return features_by_flow

    def process_and_predict(self, features_by_flow):
        if not self.model or not self.detector or not self.scaler or not self.label_encoder or not features_by_flow:
            logger.warning("Cannot predict: model, detector, scaler, label_encoder, or features missing")
            return

        from utils import detector_queue
        from queue import Full

        for flow_id, features in features_by_flow.items():
            try:
                feature_vector = [
                    features.get('Header_Length', 0),
                    features.get('Protocol Type', 'Other'),
                    features.get('Time_To_Live', 0),
                    features.get('Rate', 0),
                    features.get('fin_flag_number', 0),
                    features.get('syn_flag_number', 0),
                    features.get('rst_flag_number', 0),
                    features.get('psh_flag_number', 0),
                    features.get('ack_flag_number', 0),
                    features.get('HTTP', 0),
                    features.get('HTTPS', 0),
                    features.get('TCP', 0),
                    features.get('UDP', 0),
                    features.get('ICMP', 0),
                    features.get('Tot sum', 0),
                    features.get('Min', 0),
                    features.get('Max', 0),
                    features.get('AVG', 0),
                    features.get('Std', 0),
                    features.get('Tot size', 0),
                    features.get('IAT', 0),
                    features.get('Number', 0),
                    0  # Placeholder for the 23rd feature
                ]
                
                protocol = features.get('Protocol Type', 'Other')
                if protocol not in ['TCP', 'UDP', 'ICMP', 'Other']:
                    protocol = 'Other'
                feature_vector[1] = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'Other': 0}.get(protocol, 0)
                
                feature_names = [
                    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
                    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
                    'ack_flag_number', 'HTTP', 'HTTPS', 'TCP', 'UDP', 'ICMP',
                    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
                    'Placeholder'  # Name for the 23rd feature
                ]
                feature_vector_df = pd.DataFrame([feature_vector], columns=feature_names)
                
                # Convert DataFrame to NumPy array to avoid feature names warning
                feature_vector_array = feature_vector_df.values
                
                feature_vector_scaled = self.scaler.transform(feature_vector_array)
                
                attack_prob = self.model.predict_proba(feature_vector_scaled)[0]
                y_pred_enc = self.model.predict(feature_vector_scaled)[0]
                attack_type = self.label_encoder.inverse_transform([y_pred_enc])[0]
                
                is_attack = attack_type != 'benign'
                priority = 1 if is_attack else 0
                
                features['attack_type'] = attack_type
                
                logger.info(f"Processed flow {flow_id}: Src={features['source ip']}, Dst={features['destination ip']}, "
                           f"Rate={features['Rate']:.2f}, Attack prob={attack_prob.max():.2f}, "
                           f"Attack type={attack_type}, Is attack={is_attack}")
                
                try:
                    detector_queue.put((features, is_attack, priority), block=False)
                except Full:
                    logger.warning(f"Queue full, dropping flow {flow_id}")
                    detector_queue.get()
                    detector_queue.put((features, is_attack, priority), block=False)
            except Exception as e:
                logger.error(f"Error predicting for flow {flow_id}: {e}", exc_info=True)

    def start_capture(self):
        try:
            bpf_filter = f"not host {self.local_ip}"
            logger.info(f"Starting packet capture on {self.interface} with filter: {bpf_filter}")
            sniff(iface=self.interface, prn=self.packet_callback, filter=bpf_filter, store=0, 
                  stop_filter=lambda x: not self.running)
        except Exception as e:
            logger.error(f"Error in start_capture: {e}", exc_info=True)

    def shutdown(self):
        self.running = False

def run_traffic_analyzer(analyzer):
    try:
        logger.info(f"Starting TrafficAnalyzer on interface: {analyzer.interface}")
        analyzer.start_capture()
    except Exception as e:
        logger.error(f"Error in run_traffic_analyzer: {e}", exc_info=True)