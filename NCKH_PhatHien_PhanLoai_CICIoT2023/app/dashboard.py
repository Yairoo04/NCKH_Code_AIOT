from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time

def setup_dashboard(flask_app, detector, config):
    dash_app = Dash(__name__, server=flask_app, url_base_pathname='/dashboard/')
    
    dash_app.layout = html.Div([
        html.Div([
            html.H1("DDoS Detection Dashboard", className="dashboard-header"),
            html.Div(id="last-updated", className="last-updated"),
            html.A("Back to Home", href="/", className="back-to-home-link")
        ], className="header-container"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.H3("Network Status"), 
                    html.Div(id='status-indicator', className='status-indicator')
                ], className='panel status-panel'),
                
                html.Div([
                    html.H3("Attack Statistics"), 
                    html.Div(id='attack-stats', className='attack-stats-container')
                ], className='panel stats-panel'),
                
                html.Div([
                    html.H3("Top Source IPs"), 
                    html.Div(id='top-ips', className='top-ips-container')
                ], className='panel ips-panel'),
                
                html.Div([
                    html.H3("Should Blocked IPs"),
                    html.Div(id='blocked-ips', className='blocked-ips-container')
                ], className='panel blocked-ips-panel')
            ], className='left-column'),
            
            html.Div([
                html.Div([
                    html.H3("Traffic Monitor"), 
                    dcc.Graph(id='traffic-graph')
                ], className='panel graph-panel'),
                
                html.Div([
                    html.H3("Packet Flow Heatmap"), 
                    dcc.Graph(id='heatmap-graph')
                ], className='panel heatmap-panel'),
                
                html.Div([
                    html.H3("Feature Importance"), 
                    dcc.Graph(id='features-graph')
                ], className='panel features-panel')
            ], className='center-column'),
            
            html.Div([
                html.Div([
                    html.H3("DDoS Alerts"), 
                    html.Div(id='alerts-container', className='alerts-container scrollable')
                ], className='panel alerts-panel')
            ], className='right-column')
        ], className='dashboard-content'),
        
        dcc.Interval(
            id='interval-component',
            interval=config.dashboard_update_interval * 1000,
            n_intervals=0
        )
    ], className='dashboard-container')

    @dash_app.callback(
        [Output('status-indicator', 'children'), 
         Output('traffic-graph', 'figure'),
         Output('alerts-container', 'children'), 
         Output('attack-stats', 'children'),
         Output('top-ips', 'children'), 
         Output('heatmap-graph', 'figure'),
         Output('features-graph', 'figure'), 
         Output('last-updated', 'children'),
         Output('blocked-ips', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        try:
            status_style = {
                'padding': '20px',
                'borderRadius': '5px',
                'textAlign': 'center',
                'fontWeight': 'bold',
                'fontSize': '18px'
            }
            alerts = detector.get_alerts(count=20)
            current_time = time.time()
            
            display_attack = False
            if alerts:
                last_alert_time = time.strptime(alerts[-1]['time'], '%H:%M:%S')
                last_alert_timestamp = time.mktime(time.localtime(current_time)) - time.mktime(last_alert_time)
                display_attack = last_alert_timestamp < 60

            if detector.current_status == "Normal" or (not alerts and last_alert_timestamp > 60):
                status_color = "#2ecc71"
                status_text = "NORMAL"
            else:
                status_color = "#e74c3c"
                status_text = detector.current_status

            status = html.Div(
                status_text, 
                style={**status_style, 'backgroundColor': status_color, 'color': 'white'}
            )

            recent_data = detector.get_recent_data()
            traffic_fig = create_traffic_figure(recent_data)
            
            ip_attack_counts = detector.get_ip_attack_counts()

            alert_items = []
            for alert in reversed(alerts[-10:]):
                ip = detector.extract_ip_from_message(alert['message'])
                attack_count = ip_attack_counts.get(ip, 0) if ip else 0
                alert_items.append(
                    html.Div([
                        html.Span(f"[{alert['time']}]", className="alert-time"),
                        html.Span("🚫 " if "Blocked" in alert['mitigation'] else "⚠️ ", className="alert-icon"),
                        html.Span(f"{alert['message']}", className="alert-message"),
                        html.Span(f"Attacks: {attack_count}", className="alert-count"),
                        html.Span(f"Mitigation: {alert['mitigation']}", className="alert-mitigation")
                    ], className=f"alert-item {'severe' if 'Blocked' in alert['mitigation'] else 'warning'} new-alert")
                )

            attack_stats = detector.get_attack_stats()
            stats_items = [
                html.Div([
                    html.Div("Total Attacks", className="stat-label"),
                    html.Div(str(attack_stats['total_attacks']), className="stat-value")
                ], className="stat-item"),
                html.Div([
                    html.Div("Should Blocked IPs", className="stat-label"),
                    html.Div(str(attack_stats['blocked_ips']), className="stat-value")
                ], className="stat-item"),
                html.Div([
                    html.Div("Last Attack", className="stat-label"),
                    html.Div(
                        str(attack_stats['last_attack'] if attack_stats['last_attack'] else "N/A"), 
                        className="stat-value small-text")
                ], className="stat-item"), 
                html.Div([
                    html.Div("Blocked IP List", className="stat-label"),
                    html.Div(
                        ", ".join(attack_stats['blocked_ips_list']) if attack_stats['blocked_ips_list'] else "None", 
                        className="stat-value small-text")
                ], className="stat-item")
            ]

            top_ips = detector.get_top_ips()
            ip_items = [
                html.Div([
                    html.Div(
                        f"{i+1}. {'🚫 ' if ip_data['ip'] in detector.blocked_ips else ''}{ip_data['ip']}", 
                        className=f"ip-address {'blocked-ip' if ip_data['ip'] in detector.blocked_ips else ''}"
                    ),
                    html.Div(f"{ip_data['count']} connections", className="ip-count")
                ], className="ip-item") 
                for i, ip_data in enumerate(top_ips)
            ]

            blocked_ips = list(detector.blocked_ips)
            blocked_ip_items = [
                html.Div(f"{i+1}. {ip}", className="blocked-ip-item") 
                for i, ip in enumerate(blocked_ips)
            ] if blocked_ips else [html.Div("No IPs blocked yet", className="no-blocked-text")]

            heatmap_fig = create_heatmap_figure(recent_data)
            features_fig = create_features_figure(detector)

            last_updated = html.Div(
                f"Last updated: {time.strftime('%H:%M:%S')}",
                className="update-time"
            )

            return (
                status, traffic_fig, alert_items, stats_items, 
                ip_items, heatmap_fig, features_fig, last_updated, blocked_ip_items
            )
            
        except Exception as e:
            print(f"Error in update_dashboard: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(title='Error loading data')
            return (
                html.Div("ERROR", style={'backgroundColor': '#e74c3c', 'color': 'white'}),
                empty_fig, [], [], [], empty_fig, empty_fig,
                html.Div(f"Error: {time.strftime('%H:%M:%S')}"), []
            )

    def create_traffic_figure(recent_data):
        traffic_fig = go.Figure()
        traffic_fig.add_hline(y=50000, line_dash="dash", line_color="red", annotation_text="Bytes/s Threshold")
        traffic_fig.add_hline(y=1000, line_dash="dash", line_color="orange", annotation_text="Rate Threshold")
        
        if not recent_data.empty and len(recent_data) > 1:
            timestamps = [
                time.strftime('%H:%M:%S', time.localtime(ts)) 
                for ts in recent_data['timestamp']
                if time.time() - ts <= config.timestamp_display_duration
            ]
            if timestamps:
                valid_data = recent_data[
                    (recent_data['timestamp'] >= time.time() - config.timestamp_display_duration)
                ]
                traffic_fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=valid_data['Tot size'],
                    mode='lines', 
                    name='Bytes/s', 
                    line=dict(color='#3498db', width=2)
                ))
                traffic_fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=valid_data['Rate'],
                    mode='lines', 
                    name='Packets/s', 
                    line=dict(color='#2ecc71', width=2)
                ))
                if 'is_attack' in valid_data.columns:
                    attack_points = valid_data[valid_data['is_attack'].astype(bool)]
                    if not attack_points.empty:
                        attack_timestamps = [
                            time.strftime('%H:%M:%S', time.localtime(ts)) 
                            for ts in attack_points['timestamp']
                        ]
                        attack_types = attack_points['attack_type']
                        traffic_fig.add_trace(go.Scatter(
                            x=attack_timestamps, 
                            y=attack_points['Rate'], 
                            mode='markers', 
                            marker=dict(size=12, color='red', symbol='x'), 
                            name='Attack Points',
                            text=attack_types,
                            hoverinfo='text+y+x'
                        ))
        
        traffic_fig.update_layout(
            title='Network Traffic',
            xaxis_title='Time', 
            yaxis_title='Volume (log scale)',
            yaxis=dict(type='log'),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(240,240,240,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=300
        )
        return traffic_fig

    def create_heatmap_figure(recent_data):
        heatmap_fig = go.Figure()
        
        if not recent_data.empty and len(recent_data) > 5:
            src_ips = recent_data['source ip'].value_counts().head(10).index.tolist()
            dst_ips = recent_data['destination ip'].value_counts().head(10).index.tolist()
            
            z_values = [
                [recent_data[(recent_data['source ip'] == src) & 
                            (recent_data['destination ip'] == dst)]['Rate'].sum() 
                for dst in dst_ips] 
                for src in src_ips
            ]
            
            hover_text = [
                [f"Src: {src}<br>Dst: {dst}<br>Rate: {z_values[i][j]:.2f}<br>{'BLOCKED' if src in detector.blocked_ips else ''}"
                for j, dst in enumerate(dst_ips)]
                for i, src in enumerate(src_ips)
            ]
            
            heatmap_fig.add_trace(go.Heatmap(
                z=z_values, 
                x=dst_ips, 
                y=src_ips, 
                colorscale='Viridis',
                text=hover_text,
                hoverinfo='text',
                colorbar=dict(title='Packets/s')
            ))
            
            heatmap_fig.update_layout(
                title='Source to Destination Traffic (Top 10 IPs)',
                xaxis_title='Destination IP', 
                yaxis_title='Source IP',
                margin=dict(l=50, r=20, t=40, b=50),
                plot_bgcolor='rgba(240,240,240,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50'),
                height=400,
                xaxis=dict(tickangle=45),
                yaxis=dict(tickangle=0)
            )
        else:
            heatmap_fig.update_layout(
                title='Insufficient data for heatmap',
                height=400
            )
        
        return heatmap_fig

    def create_features_figure(detector):
        features_fig = go.Figure()
        
        result = detector.get_feature_importance() 

        if result is None:
            features_fig.update_layout(
                title='Feature Importance in Attack Detection',
                xaxis_title='Importance Score',
                margin=dict(l=150, r=20, t=40, b=20),
                plot_bgcolor='rgba(240,240,240,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50'),
                height=300,
                xaxis=dict(range=[0, 1])
            )
            return features_fig
        
        features, importance_dict = result
        
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_features = [item[0] for item in sorted_items]
        sorted_importance = [item[1] for item in sorted_items]
        
        colors = ['rgba(231, 76, 60, 0.8)' if val > 0.6 else 'rgba(52, 152, 219, 0.8)' 
                 for val in sorted_importance]
        
        features_fig.add_trace(go.Bar(
            y=sorted_features, 
            x=sorted_importance, 
            orientation='h', 
            marker=dict(
                color=colors, 
                line=dict(color='rgba(0, 0, 0, 0.2)', width=1)
            )
        ))
        
        features_fig.update_layout(
            title='Feature Importance in Attack Detection',
            xaxis_title='Importance Score',
            margin=dict(l=150, r=20, t=40, b=20),
            plot_bgcolor='rgba(240,240,240,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=300,
            xaxis=dict(range=[0, 1])
        )
        
        return features_fig

    return dash_app