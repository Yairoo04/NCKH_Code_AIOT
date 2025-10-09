import time
from threading import Thread
from collections import defaultdict, deque
import logging
import socket
from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
import numpy as np
import torch
import joblib
import os
import psutil
from queue import Full
from utils import get_default_interface, logger, attack_logger, no_attack_logger

try:
    from tab_transformer_pytorch import TabTransformer
except Exception:
    logger.warning("Không import được TabTransformer (tab_transformer_pytorch). Hãy kiểm tra package.")

def _get_ipv4_of_interface(name: str) -> str:
    try:
        for addr in psutil.net_if_addrs().get(name, []):
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                return addr.address
    except Exception:
        pass
    return "127.0.0.1"

class TrafficAnalyzer:
    EMA_ALPHA = 0.5
    FLOW_HISTORY_LEN = 5
    FLOW_FREQ_WINDOW = 10
    RULE_PPS_ALERT = 6_000
    RULE_BPS_ALERT = 3_000_000
    RULE_PPS_SUS = 2_000
    RULE_BPS_SUS = 600_000
    CPU_HOT = 80.0
    MODEL_ATTACK_PROB = 0.95  
    MIN_DURATION = 1.0  
    IP_FREQ_ATTACK_THRESHOLD = 15 

    def __init__(self, interface, window_size=1, detector=None):
        self.interface = interface or get_default_interface() 
        self.window_size = max(0.2, float(window_size))
        self.flows = defaultdict(list)
        self.start_time = time.time()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.categorical_cols = []
        self.cont_cols = []
        self.model, self.scaler, self.label_encoder, self.categorical_cols, self.cont_cols = self.load_model()
        self.detector = detector
        self.running = True
        self.local_ip = _get_ipv4_of_interface(self.interface)
        logger.info(f"Interface selected: {self.interface} | Local IP: {self.local_ip}")
        self.flow_stats = defaultdict(lambda: {"ema_pps": 0.0, "ema_bps": 0.0})
        self.flow_history = defaultdict(lambda: {"packets": deque(maxlen=self.FLOW_HISTORY_LEN),
                                                 "bytes": deque(maxlen=self.FLOW_HISTORY_LEN)})
        self.src_ip_flow_count = defaultdict(lambda: {"count": 0, "last_time": time.time()})
        self.src_ip_packet_count = defaultdict(int) 
    def load_model(self):
        try:
            from config import Config
            config = Config()
            model_path = config.model_path
            scaler_path = config.scaler_path
            encoder_path = config.encoder_path
            categories_path = "models/categories.pkl"
            num_continuous_path = "models/num_continuous.pkl"
            for p in [model_path, scaler_path, encoder_path, categories_path, num_continuous_path]:
                if not os.path.exists(p):
                    logger.warning(f"Không tìm thấy tệp: {p} - Fallback to rules-only mode.")
                    return None, None, None, None, None
                if os.path.getsize(p) == 0:
                    logger.warning(f"Tệp {p} rỗng - Fallback to rules-only mode.")
                    return None, None, None, None, None
            logger.info("Đang tải scaler/encoder/categories/num_continuous...")
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
            categories = joblib.load(categories_path)
            num_continuous = joblib.load(num_continuous_path)
            categorical_cols = ['Protocol Type'] if categories else []
            cont_cols = [
                'Header_Length','Time_To_Live','Rate',
                'fin_flag_number','syn_flag_number','rst_flag_number',
                'psh_flag_number','ack_flag_number',
                'ack_count','syn_count','fin_count','rst_count',
                'HTTP','HTTPS','DNS','Telnet','SSH','TCP','UDP','ICMP',
                'Tot sum','Min','Max','AVG','Std','Tot size',
                'IAT','Number','Variance',
                'flow packets/s','flow bytes/s'
            ]
            num_classes = len(getattr(label_encoder, 'classes_', [0,1]))
            model = TabTransformer(
                categories=categories,
                num_continuous=num_continuous,
                dim=128, dim_out=num_classes, depth=6, heads=8,
                attn_dropout=0.3, ff_dropout=0.3,
                mlp_hidden_mults=(4,2),
                mlp_act=torch.nn.ReLU()
            )
            logger.info(f"Đang load state_dict TabTransformer từ {model_path} ...")
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Đã load TabTransformer model thành công")
            return model, scaler, label_encoder, categorical_cols, cont_cols
        except Exception as e:
            logger.error(f"Lỗi khi load model/scaler/encoder: {e}", exc_info=True)
            return None, None, None, None, None

    def packet_callback(self, packet):
        try:
            if not self.running or IP not in packet:
                return
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            proto = 'Other'
            if TCP in packet: proto = 'TCP'
            elif UDP in packet: proto = 'UDP'
            elif ICMP in packet: proto = 'ICMP'
            src_port = (packet[TCP].sport if TCP in packet else
                        packet[UDP].sport if UDP in packet else 0)
            dst_port = (packet[TCP].dport if TCP in packet else
                        packet[UDP].dport if UDP in packet else 0)
            flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
            pkt_len = int(len(packet))
            ip_hl = (packet[IP].ihl * 4) if IP in packet and getattr(packet[IP], 'ihl', None) else 20
            tcp_hl = (packet[TCP].dataofs * 4) if TCP in packet and getattr(packet[TCP], 'dataofs', None) else 0
            flags = int(packet[TCP].flags) if TCP in packet else 0
            ttl = int(packet[IP].ttl) if IP in packet else 0
            pkt_data = {
                'timestamp': time.time(),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'packet_length': pkt_len,
                'ip_header_len': ip_hl,
                'tcp_header_len': tcp_hl,
                'ttl': ttl,
                'flags': flags,
                'src_port': src_port,
                'dst_port': dst_port,
                'is_http': int(src_port == 80 or dst_port == 80),
                'is_https': int(src_port == 443 or dst_port == 443),
                'protocol': proto,
            }
            self.flows[flow_id].append(pkt_data)
            # Track total packets per IP
            self.src_ip_packet_count[src_ip] += 1
            if len(self.flows[flow_id]) >= 60 or (time.time() - self.start_time >= self.window_size):
                self._process_flows_for_single_flow(flow_id)
                self.start_time = time.time()
        except Exception as e:
            logger.error(f"Error in packet_callback: {e}", exc_info=True)

    def _process_flows_for_single_flow(self, flow_id):
        if flow_id not in self.flows:
            return
        features_by_flow = self.extract_features_by_flow(flow_id)
        self.process_and_predict(features_by_flow)
        try:
            del self.flows[flow_id]
        except Exception:
            pass

    @staticmethod
    def _flag_counts(flags_list):
        fin = sum(1 for f in flags_list if f & 0x01)
        syn = sum(1 for f in flags_list if f & 0x02)
        rst = sum(1 for f in flags_list if f & 0x04)
        psh = sum(1 for f in flags_list if f & 0x08)
        ack = sum(1 for f in flags_list if f & 0x10)
        return fin, syn, rst, psh, ack

    def extract_features_by_flow(self, flow_id):
        features_by_flow = {}
        packets = self.flows[flow_id]
        if not packets:
            return features_by_flow
        src_ip = packets[0]['src_ip']
        dst_ip = packets[0]['dst_ip']
        proto = packets[0]['protocol']
        total_packets = len(packets)
        ts = [p['timestamp'] for p in packets]
        packet_sizes = [p['packet_length'] for p in packets]
        ttl_values = [p['ttl'] for p in packets]
        header_lengths = [p['ip_header_len'] + p['tcp_header_len'] for p in packets]
        flags_list = [p['flags'] for p in packets]
        fin_count, syn_count, rst_count, psh_count, ack_count = self._flag_counts(flags_list)
        tot_size = sum(packet_sizes)
        duration = max(ts[-1] - ts[0], self.MIN_DURATION) if ts else self.MIN_DURATION  # Prevent inflated rates
        flow_pps = total_packets / duration
        flow_bps = tot_size / duration
        flow_stats = self.flow_stats[flow_id]
        flow_stats["ema_pps"] = self.EMA_ALPHA * flow_pps + (1 - self.EMA_ALPHA) * flow_stats["ema_pps"]
        flow_stats["ema_bps"] = self.EMA_ALPHA * flow_bps + (1 - self.EMA_ALPHA) * flow_stats["ema_bps"]
        history = self.flow_history[flow_id]
        history["packets"].append(total_packets)
        history["bytes"].append(tot_size)
        avg_pps = np.mean(history["packets"]) / duration if history["packets"] else flow_pps
        avg_bps = np.mean(history["bytes"]) / duration if history["bytes"] else flow_bps
        def ratio(count): return count / total_packets if total_packets else 0.0
        feature = {
            'flow id': flow_id,
            'source ip': src_ip,
            'destination ip': dst_ip,
            'Protocol Type': proto,
            'Header_Length': np.mean(header_lengths),
            'Time_To_Live': np.mean(ttl_values),
            'Rate': flow_stats["ema_pps"],
            'fin_flag_number': ratio(fin_count),
            'syn_flag_number': ratio(syn_count),
            'rst_flag_number': ratio(rst_count),
            'psh_flag_number': ratio(psh_count),
            'ack_flag_number': ratio(ack_count),
            'ack_count': ack_count,
            'syn_count': syn_count,
            'fin_count': fin_count,
            'rst_count': rst_count,
            'HTTP': int(any(p.get('is_http', 0) for p in packets)),
            'HTTPS': int(any(p.get('is_https', 0) for p in packets)),
            'DNS': int(any(p['src_port'] == 53 or p['dst_port'] == 53 for p in packets)),
            'Telnet': int(any(p['src_port'] == 23 or p['dst_port'] == 23 for p in packets)),
            'SSH': int(any(p['src_port'] == 22 or p['dst_port'] == 22 for p in packets)),
            'TCP': int(proto == 'TCP'),
            'UDP': int(proto == 'UDP'),
            'ICMP': int(proto == 'ICMP'),
            'Tot sum': sum(packet_sizes),
            'Min': min(packet_sizes),
            'Max': max(packet_sizes),
            'AVG': (tot_size / total_packets) if total_packets else 0.0,
            'Std': float(np.std(packet_sizes)) if len(packet_sizes) > 1 else 0.0,
            'Tot size': tot_size,
            'IAT': float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.0,
            'Number': total_packets,
            'Variance': float(np.var(packet_sizes)) if len(packet_sizes) > 1 else 0.0,
            'flow packets/s': float(flow_stats["ema_pps"]),
            'flow bytes/s': float(flow_stats["ema_bps"]),
            'attack_type': None
        }
        features_by_flow[flow_id] = feature
        now = time.time()
        if now - self.src_ip_flow_count[src_ip]["last_time"] > self.FLOW_FREQ_WINDOW:
            self.src_ip_flow_count[src_ip] = {"count": 0, "last_time": now}
        self.src_ip_flow_count[src_ip]["count"] += 1
        return features_by_flow

    def process_and_predict(self, features_by_flow):
        if not self.model or not self.scaler or not self.label_encoder:
            logger.warning("Cannot predict: missing model/scaler/label_encoder")
            return
        from utils import detector_queue
        protocol_map = {'Other': 0, 'TCP': 6, 'UDP': 17, 'ICMP': 1}
        scaler_feature_names = list(getattr(self.scaler, "feature_names_in_", []))
        expected_cont_features = scaler_feature_names if scaler_feature_names else list(self.cont_cols)
        for flow_id, feat in features_by_flow.items():
            try:
                df = pd.DataFrame([feat])
                if 'Protocol Type' in df.columns:
                    df['Protocol Type'] = df['Protocol Type'].map(protocol_map).fillna(0).astype(int)
                for c in expected_cont_features:
                    if c not in df.columns:
                        df[c] = 0.0
                X_cont_df = df[expected_cont_features].astype(float)
                X_cont_scaled = self.scaler.transform(X_cont_df)
                X_cont_tensor = torch.tensor(X_cont_scaled, dtype=torch.float32)
                if self.categorical_cols:
                    X_cat = df[self.categorical_cols].values.astype(int)
                    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long)
                else:
                    X_cat_tensor = torch.empty((X_cont_tensor.shape[0], 0), dtype=torch.long)
                with torch.no_grad():
                    outputs = self.model(X_cat_tensor, X_cont_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    max_prob = float(np.max(probs, axis=1)[0])
                    pred = int(np.argmax(probs, axis=1)[0])
                try:
                    attack_type = str(self.label_encoder.inverse_transform([pred])[0])
                except Exception:
                    attack_type = str(pred)
                
                if 'tcp' in attack_type.lower() and feat.get('UDP', 0) == 1:
                    attack_type = 'ddos-udp_flood'
                    logger.info(f"Overrode TCP pred to UDP flood for {flow_id}")
                elif 'tcp' in attack_type.lower() and feat.get('ICMP', 0) == 1:
                    attack_type = 'ddos-icmp_flood'
                    logger.info(f"Overrode TCP pred to ICMP flood for {flow_id}")
                
                pps = float(feat.get('flow packets/s', 0.0))
                bps = float(feat.get('flow bytes/s', 0.0))
                src_ip = feat.get('source ip', 'unknown')
                flow_freq = self.src_ip_flow_count[src_ip]["count"]
                ip_total_packets = self.src_ip_packet_count[src_ip]  
                cpu_usage = psutil.cpu_percent(interval=0.0)
                total_packets = feat.get('Number', 0)
                rule_alert = (pps >= self.RULE_PPS_ALERT) or (bps >= self.RULE_BPS_ALERT)
                rule_sus = (pps >= self.RULE_PPS_SUS) or (bps >= self.RULE_BPS_SUS)
                model_attack = (max_prob >= self.MODEL_ATTACK_PROB and attack_type.lower() != 'benign')
                is_attack = False
                if rule_alert:
                    is_attack = True
                elif model_attack and (rule_sus or flow_freq > self.IP_FREQ_ATTACK_THRESHOLD or ip_total_packets > 20): 
                    is_attack = True
                elif cpu_usage >= self.CPU_HOT and (model_attack or rule_sus):
                    is_attack = True
                if total_packets < 3 and ip_total_packets < 10 and not rule_alert and cpu_usage < self.CPU_HOT:
                    is_attack = False
                final_attack_type = attack_type if is_attack else 'benign'
                final_prob = max_prob if is_attack else 1.0
                log_message = (
                    f"[{flow_id}] Src={feat.get('source ip')} Dst={feat.get('destination ip')} "
                    f"PPS(ema)={pps:.1f} BPS(ema)={bps:.0f} Freq={flow_freq} IP_Pkts={ip_total_packets} Packets={total_packets} "
                    f"Pred={attack_type} P={max_prob:.3f} CPU={cpu_usage:.1f}% -> "
                    f"{'ATTACK' if is_attack else 'OK'}"
                )
                logger.info(log_message)
                feat['attack_type'] = final_attack_type
                feat['probability'] = float(final_prob)
                priority = 1 if is_attack else 0
                try:
                    detector_queue.put((feat, is_attack, priority, final_prob), block=False)
                except Full:
                    logger.warning(f"Queue full, dropping {flow_id}")
                    try:
                        detector_queue.get_nowait()
                        detector_queue.put((feat, is_attack, priority, final_prob), block=False)
                    except Exception as e:
                        logger.error(f"Failed to handle queue full for {flow_id}: {e}")
            except Exception as e:
                logger.error(f"Error predicting for flow {flow_id}: {e}", exc_info=True)

    def start_capture(self):
        try:
            bpf_filter = f"dst host {self.local_ip}"
            logger.info(f"Starting capture on {self.interface} filter: {bpf_filter}")
            sniff(iface=self.interface,
                  prn=self.packet_callback,
                  filter=bpf_filter,
                  store=0,
                  stop_filter=lambda _: not self.running)
        except Exception as e:
            logger.error(f"Error in start_capture: {e}", exc_info=True)

    def shutdown(self):
        self.running = False
        self.src_ip_packet_count.clear()

def run_traffic_analyzer(analyzer: "TrafficAnalyzer"):
    try:
        logger.info(f"Starting TrafficAnalyzer on interface: {analyzer.interface}")
        analyzer.start_capture()
    except Exception as e:
        logger.error(f"Error in run_traffic_analyzer: {e}", exc_info=True)