"""
Azure ML Model Monitoring and Data Drift Detection
Comprehensive monitoring solution for 3D Latent Diffusion Model
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Azure ML imports
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ModelMonitor,
    MonitoringTarget,
    MonitorDefinition,
    MonitorSchedule,
    AlertNotification
)
from azure.identity import DefaultAzureCredential
import yaml

# Monitoring and alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitoring:
    """Comprehensive model monitoring for 3D LDM"""
    
    def __init__(self, config_path: str = None):
        """Initialize monitoring system"""
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "../azure/workspace_config.yml"
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config['subscription_id'],
            resource_group_name=self.config['resource_group'],
            workspace_name=self.config['workspace_name']
        )
        
        self.monitoring_config = self._load_monitoring_config()
        logger.info(f"✅ Monitoring system initialized for workspace: {self.config['workspace_name']}")
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "data_drift_threshold": 0.3,
            "performance_threshold": 0.8,
            "latency_threshold_ms": 10000,
            "error_rate_threshold": 0.05,
            "alert_email": "admin@yourcompany.com",
            "monitoring_frequency": "daily",
            "retention_days": 30
        }
        
        # Try to load custom config
        config_file = Path(__file__).parent / "monitoring_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        return default_config
    
    def setup_data_drift_monitoring(self, 
                                   endpoint_name: str,
                                   baseline_dataset: str,
                                   target_dataset: str = None) -> ModelMonitor:
        """Set up data drift monitoring for the deployed model"""
        
        logger.info(f"Setting up data drift monitoring for endpoint: {endpoint_name}")
        
        # Define monitoring target
        monitoring_target = MonitoringTarget(
            ml_task="regression",  # Adjust based on your model type
            endpoint_deployment_id=f"azureml://subscriptions/{self.config['subscription_id']}/resourceGroups/{self.config['resource_group']}/providers/Microsoft.MachineLearningServices/workspaces/{self.config['workspace_name']}/onlineEndpoints/{endpoint_name}/deployments/default"
        )
        
        # Create monitor definition
        monitor_definition = MonitorDefinition(
            compute_target="serverless",
            monitoring_target=monitoring_target,
            alert_notification=AlertNotification(
                emails=[self.monitoring_config["alert_email"]]
            )
        )
        
        # Create monitor schedule
        monitor_schedule = MonitorSchedule(
            name=f"{endpoint_name}-data-drift-monitor",
            trigger={
                "type": "recurrence",
                "frequency": self.monitoring_config["monitoring_frequency"],
                "interval": 1
            },
            create_monitor=monitor_definition
        )
        
        # Create the monitor
        monitor = self.ml_client.schedules.begin_create_or_update(monitor_schedule)
        
        logger.info(f"✅ Data drift monitoring setup completed for {endpoint_name}")
        return monitor
    
    def setup_performance_monitoring(self, endpoint_name: str):
        """Set up performance monitoring for model endpoint"""
        
        logger.info(f"Setting up performance monitoring for endpoint: {endpoint_name}")
        
        # This would typically integrate with Azure Monitor/Application Insights
        performance_config = {
            "endpoint_name": endpoint_name,
            "metrics": [
                "request_latency",
                "requests_per_second", 
                "error_rate",
                "cpu_utilization",
                "memory_utilization"
            ],
            "thresholds": {
                "latency_ms": self.monitoring_config["latency_threshold_ms"],
                "error_rate": self.monitoring_config["error_rate_threshold"]
            },
            "alert_rules": [
                {
                    "metric": "request_latency",
                    "condition": f"average > {self.monitoring_config['latency_threshold_ms']}",
                    "severity": "warning"
                },
                {
                    "metric": "error_rate", 
                    "condition": f"average > {self.monitoring_config['error_rate_threshold']}",
                    "severity": "critical"
                }
            ]
        }
        
        # Save performance monitoring config
        config_file = Path(__file__).parent / f"{endpoint_name}_performance_config.json"
        with open(config_file, 'w') as f:
            json.dump(performance_config, f, indent=2)
        
        logger.info(f"✅ Performance monitoring configuration saved for {endpoint_name}")
        return performance_config
    
    def check_model_performance(self, 
                               endpoint_name: str,
                               time_range_hours: int = 24) -> Dict[str, Any]:
        """Check model performance metrics"""
        
        logger.info(f"Checking performance for endpoint: {endpoint_name}")
        
        # This is a placeholder for actual metric collection
        # In practice, you would query Azure Monitor or Application Insights
        
        # Simulate performance metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Mock metrics (replace with actual Azure Monitor queries)
        metrics = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "request_count": np.random.randint(100, 1000),
            "average_latency_ms": np.random.uniform(1000, 5000),
            "error_rate": np.random.uniform(0.01, 0.1),
            "p95_latency_ms": np.random.uniform(2000, 8000),
            "p99_latency_ms": np.random.uniform(5000, 12000),
            "throughput_rps": np.random.uniform(1, 10),
            "cpu_utilization": np.random.uniform(20, 80),
            "memory_utilization": np.random.uniform(30, 90)
        }
        
        # Check against thresholds
        alerts = []
        if metrics["average_latency_ms"] > self.monitoring_config["latency_threshold_ms"]:
            alerts.append({
                "type": "latency",
                "severity": "warning",
                "message": f"Average latency ({metrics['average_latency_ms']:.2f}ms) exceeds threshold ({self.monitoring_config['latency_threshold_ms']}ms)"
            })
        
        if metrics["error_rate"] > self.monitoring_config["error_rate_threshold"]:
            alerts.append({
                "type": "error_rate",
                "severity": "critical", 
                "message": f"Error rate ({metrics['error_rate']:.3f}) exceeds threshold ({self.monitoring_config['error_rate_threshold']})"
            })
        
        result = {
            "endpoint_name": endpoint_name,
            "metrics": metrics,
            "alerts": alerts,
            "status": "healthy" if not alerts else "unhealthy",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"⚠️  Alert: {alert['message']}")
        
        return result
    
    def detect_data_drift(self, 
                         baseline_data: np.ndarray,
                         current_data: np.ndarray,
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect data drift between baseline and current data"""
        
        logger.info("Detecting data drift...")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(baseline_data.shape[1])]
        
        # Calculate drift metrics for each feature
        drift_results = []
        
        for i, feature_name in enumerate(feature_names):
            baseline_feature = baseline_data[:, i]
            current_feature = current_data[:, i]
            
            # Statistical tests for drift detection
            from scipy import stats
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = stats.ks_2samp(baseline_feature, current_feature)
            
            # Population stability index (PSI)
            psi = self._calculate_psi(baseline_feature, current_feature)
            
            # Determine if drift detected
            drift_detected = (ks_p_value < 0.05) or (psi > self.monitoring_config["data_drift_threshold"])
            
            feature_result = {
                "feature_name": feature_name,
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p_value),
                "psi_score": float(psi),
                "drift_detected": drift_detected,
                "drift_severity": self._get_drift_severity(psi)
            }
            
            drift_results.append(feature_result)
            
            if drift_detected:
                logger.warning(f"⚠️  Data drift detected for feature: {feature_name} (PSI: {psi:.3f})")
        
        # Overall drift summary
        total_features = len(feature_names)
        drifted_features = sum(1 for result in drift_results if result["drift_detected"])
        overall_drift_score = drifted_features / total_features
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_features": total_features,
            "drifted_features": drifted_features,
            "overall_drift_score": overall_drift_score,
            "drift_detected": overall_drift_score > 0.2,  # 20% threshold
            "feature_results": drift_results
        }
        
        return summary
    
    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        
        # Create bins based on baseline data
        bin_edges = np.histogram_bin_edges(baseline, bins=bins)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to get proportions
        baseline_prop = baseline_dist / np.sum(baseline_dist)
        current_prop = current_dist / np.sum(current_dist)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_prop = np.maximum(baseline_prop, epsilon)
        current_prop = np.maximum(current_prop, epsilon)
        
        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
        
        return psi
    
    def _get_drift_severity(self, psi_score: float) -> str:
        """Get drift severity based on PSI score"""
        if psi_score < 0.1:
            return "low"
        elif psi_score < 0.2:
            return "medium"
        else:
            return "high"
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert notification"""
        
        try:
            # Format alert message
            message = self._format_alert_message(alert_data)
            
            # Send email alert
            self._send_email_alert(
                subject=f"3D LDM Model Alert - {alert_data.get('type', 'Unknown')}",
                message=message,
                recipient=self.monitoring_config["alert_email"]
            )
            
            logger.info("📧 Alert sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {str(e)}")
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert message"""
        
        message = f"""
3D Latent Diffusion Model Alert

Alert Type: {alert_data.get('type', 'Unknown')}
Severity: {alert_data.get('severity', 'Unknown')}
Timestamp: {alert_data.get('timestamp', datetime.now().isoformat())}

Details:
{alert_data.get('message', 'No details available')}

Endpoint: {alert_data.get('endpoint_name', 'Unknown')}

Please investigate and take appropriate action.
        """
        
        return message.strip()
    
    def _send_email_alert(self, subject: str, message: str, recipient: str):
        """Send email alert (placeholder implementation)"""
        
        # This is a placeholder - implement with your email service
        logger.info(f"📧 Email alert would be sent to {recipient}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Message: {message}")
    
    def generate_monitoring_report(self, 
                                  endpoint_name: str,
                                  time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        logger.info(f"Generating monitoring report for {endpoint_name}")
        
        # Collect performance metrics
        performance_data = self.check_model_performance(endpoint_name, time_range_hours)
        
        # Generate report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "endpoint_name": endpoint_name,
            "time_range_hours": time_range_hours,
            "performance_metrics": performance_data["metrics"],
            "alerts": performance_data["alerts"],
            "overall_status": performance_data["status"],
            "recommendations": self._generate_recommendations(performance_data)
        }
        
        # Save report
        report_file = Path(__file__).parent / f"monitoring_report_{endpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Monitoring report saved: {report_file}")
        
        return report
    
    def _generate_recommendations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance data"""
        
        recommendations = []
        metrics = performance_data["metrics"]
        
        if metrics["average_latency_ms"] > self.monitoring_config["latency_threshold_ms"]:
            recommendations.append("Consider scaling up the deployment or optimizing model inference")
        
        if metrics["error_rate"] > self.monitoring_config["error_rate_threshold"]:
            recommendations.append("Investigate error logs and consider model retraining")
        
        if metrics["cpu_utilization"] > 80:
            recommendations.append("Consider scaling up compute resources")
        
        if metrics["memory_utilization"] > 85:
            recommendations.append("Consider increasing memory allocation or optimizing model size")
        
        if not recommendations:
            recommendations.append("System is performing within acceptable parameters")
        
        return recommendations

def main():
    """Main function for testing monitoring system"""
    
    # Initialize monitoring
    monitor = ModelMonitoring()
    
    # Example: Set up monitoring for an endpoint
    endpoint_name = "3d-ldm-production"
    
    # Set up data drift monitoring
    monitor.setup_data_drift_monitoring(
        endpoint_name=endpoint_name,
        baseline_dataset="baseline_data_v1"
    )
    
    # Set up performance monitoring
    monitor.setup_performance_monitoring(endpoint_name)
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report(endpoint_name)
    
    print("✅ Monitoring setup completed!")
    print(f"Report generated for {endpoint_name}")

if __name__ == "__main__":
    main()