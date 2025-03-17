import google.generativeai as genai
import os
from datetime import datetime, timedelta
import time
import json
import random
import string
import re
from typing import List, Dict
import argparse
import logging
import sys
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class EmailConfig:
    """Enhanced configuration for smart email generation"""
    min_word_count: int = 100  # Increased minimum for more detailed content
    max_word_count: int = 200  # Increased maximum for comprehensive reports
    min_emails: int = 1
    max_emails: int = 100
    rate_limit_delay: int = 1
    metrics_history_days: int = 90  # Track metrics history


@dataclass
class MetricsGenerator:
    """Generates realistic-looking metrics data"""
   
    def __init__(self):
        self.focus_area = 'all'
        self.threshold = 0.10  # Default threshold for anomaly detection
        self.metric_groups = {
            'performance': [
                'response_time',
                'throughput',
                'latency',
                'requests_per_second'
            ],
            'reliability': [
                'error_rate',
                'uptime',
                'availability',
                'success_rate'
            ],
            'resource': [
                'cpu_usage',
                'memory_usage',
                'disk_usage',
                'network_bandwidth'
            ]
        }
   
    def set_focus(self, focus_area: str):
        """Set the focus area for metrics generation"""
        if focus_area not in ['performance', 'reliability', 'resource', 'all']:
            raise ValueError(f"Invalid focus area: {focus_area}")
        self.focus_area = focus_area
   
    def set_threshold(self, threshold: float):
        """Set the threshold for anomaly detection"""
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
   
    def generate_time_series(self, days: int, trend: float = 0.1, volatility: float = 0.05) -> List[float]:
        """Generate a realistic time series with trend and volatility"""
        base = 100
        noise = np.random.normal(0, volatility, days)
        trend_component = np.linspace(0, trend * days, days)
        return [max(0, base + t + n) for t, n in zip(trend_component, noise)]
   
    def generate_performance_metrics(self) -> Dict[str, Dict]:
        """Generate a comprehensive set of realistic performance metrics"""
        # Select metrics based on focus area
        metrics = {}
       
        if self.focus_area == 'all':
            metric_keys = [metric for group in self.metric_groups.values() for metric in group]
        else:
            metric_keys = self.metric_groups[self.focus_area]
       
        # Default metric configurations
        metric_configs = {
            'response_time': {'target': 100, 'unit': 'ms', 'range': (80, 120)},
            'throughput': {'target': 1000, 'unit': 'req/s', 'range': (950, 1050)},
            'latency': {'target': 50, 'unit': 'ms', 'range': (40, 60)},
            'requests_per_second': {'target': 500, 'unit': 'req/s', 'range': (450, 550)},
            'error_rate': {'target': 0.1, 'unit': '%', 'range': (0.1, 0.5)},
            'uptime': {'target': 99.9, 'unit': '%', 'range': (99.5, 100)},
            'availability': {'target': 99.95, 'unit': '%', 'range': (99.8, 100)},
            'success_rate': {'target': 99.5, 'unit': '%', 'range': (98, 100)},
            'cpu_usage': {'target': 70, 'unit': '%', 'range': (60, 80)},
            'memory_usage': {'target': 75, 'unit': '%', 'range': (65, 85)},
            'disk_usage': {'target': 80, 'unit': '%', 'range': (70, 90)},
            'network_bandwidth': {'target': 800, 'unit': 'Mbps', 'range': (700, 900)}
        }
       
        # Generate metrics based on selected keys
        for metric_key in metric_keys:
            config = metric_configs[metric_key]
            current = round(random.uniform(*config['range']), 2)
           
            # Add some anomalies based on threshold
            if random.random() < self.threshold:
                # Generate a more significant deviation
                deviation = random.choice([-1, 1]) * random.uniform(0.2, 0.4)
                current = round(config['target'] * (1 + deviation), 2)
           
            metrics[metric_key] = {
                'current': current,
                'target': config['target'],
                'unit': config['unit'],
                'trend': self.generate_time_series(30,
                                                 trend=random.uniform(-0.05, 0.05),
                                                 volatility=random.uniform(0.02, 0.08))
            }
       
        return metrics


class EnhancedTestEmailMetadata:
    """Enhanced metadata generator with smart categorization"""
   
    def __init__(self):
        self.domains = ['testcompany.com', 'testing.org', 'qamail.net', 'unittest.io']
        self.departments = ['platform', 'infrastructure', 'sre', 'devops', 'performance', 'monitoring']
        self.metrics_categories = [
            "System Performance",
            "Resource Utilization",
            "Service Reliability",
            "Platform Scalability",
            "Infrastructure Efficiency"
        ]
        self.used_subjects = set()
       
    def generate_email(self) -> str:
        name = ''.join(random.choices(string.ascii_lowercase, k=8))
        department = random.choice(self.departments)
        domain = random.choice(self.domains)
        return f"{name}.{department}@{domain}"
   
    def generate_unique_subject(self, test_id: str, metrics: Dict) -> str:
        """Generate subject based on actual metrics content"""
        category = random.choice(self.metrics_categories)
       
        # Find most significant metric change
        significant_metric = max(metrics.items(),
                               key=lambda x: abs(x[1]['current'] - x[1]['target']))
       
        metric_name, metric_data = significant_metric
        performance = "Above" if metric_data['current'] > metric_data['target'] else "Below"
       
        subject = f"[TEST] {category} Alert: {metric_name.replace('_', ' ').title()} {performance} Target - {test_id}"
        subject_hash = hashlib.md5(subject.encode()).hexdigest()
       
        if subject_hash not in self.used_subjects:
            self.used_subjects.add(subject_hash)
            return subject
       
        return self.generate_unique_subject(test_id, metrics)  # Recursively try again if collision


class SmartTestEmailGenerator:
    def __init__(self, api_key: str):
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
       
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            self.metadata = EnhancedTestEmailMetadata()
            self.metrics_generator = MetricsGenerator()
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini AI: {str(e)}")
            raise


        self.test_disclaimer = """
+==================================================+
|            AUTOMATED METRICS REPORT               |
|            THIS IS A TEST EMAIL                   |
|            DO NOT REPLY OR TAKE ACTION            |
+==================================================+
Report ID: {test_id}
Generated: {timestamp}
Period: {start_date} to {end_date}
This is an automated test report. All metrics and trends
are simulated for testing purposes.
+==================================================+
"""


    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('metrics_reports.log')
            ]
        )


    def generate_test_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"METRICS-{timestamp}-{random_suffix}"


    def format_metric_table(self, metrics: Dict) -> str:
        """Create an ASCII table showing metrics performance"""
        table = [
            "+-----------------+----------+---------+----------+-----------+",
            "| Metric          | Current  | Target  | Unit     | Status    |",
            "+-----------------+----------+---------+----------+-----------+"
        ]
       
        for name, data in metrics.items():
            current = data['current']
            target = data['target']
            unit = data['unit']
           
            # Calculate status based on target
            diff_percent = (current - target) / target * 100
            if abs(diff_percent) <= 5:
                status = "NORMAL"
            elif diff_percent > 5:
                status = "HIGH"
            else:
                status = "LOW"
           
            row = f"| {name:<15} | {current:>8.2f} | {target:>7.2f} | {unit:<8} | {status:<9} |"
            table.append(row)
       
        table.append("+-----------------+----------+---------+----------+-----------+")
        return "\n".join(table)


    def generate_trend_analysis(self, metrics: Dict) -> str:
        """Generate a text-based trend analysis of the metrics"""
        analysis = []
        for metric, data in metrics.items():
            trend = data['trend']
            start_val, end_val = trend[0], trend[-1]
            change = ((end_val - start_val) / start_val) * 100
           
            direction = "improved" if change > 0 else "declined"
            analysis.append(f"* {metric.replace('_', ' ').title()}: {abs(change):.1f}% {direction} over the past 30 days")
       
        return "\n".join(analysis)


    def generate_prompt(self, metrics: Dict, word_count: int, test_id: str) -> str:
        from_email = self.metadata.generate_email()
        to_email = self.metadata.generate_email()
        timestamp = datetime.now()
        start_date = (timestamp - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = timestamp.strftime("%Y-%m-%d")
        subject = self.metadata.generate_unique_subject(test_id, metrics)
       
        metric_table = self.format_metric_table(metrics)
        trend_analysis = self.generate_trend_analysis(metrics)
       
        return f"""Generate a technical metrics report email with this structure:
                    From: {from_email}
                    To: {to_email}
                    Date: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                    Subject: {subject}


                    {self.test_disclaimer.format(
                        test_id=test_id,
                        timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        start_date=start_date,
                        end_date=end_date
                    )}


                    Executive Summary:
                    - Focus on significant metric changes
                    - Highlight critical trends
                    - Emphasize business impact


                    Current Metrics Performance:
                    {metric_table}


                    Trend Analysis:
                    {trend_analysis}


                    Requirements:
                    - Approximately {word_count} words
                    - Include specific recommendations based on metrics
                    - Focus on actionable insights
                    - Reference specific monitoring tools and systems
                    - Suggest concrete next steps for improvement
                   
                    End with a professional signature including:
                    - Technical role
                    - Department
                    - Test environment details"""


    async def generate_single_email(self, word_count: int, email_number: int) -> Dict:
        test_id = self.generate_test_id()
       
        try:
            metrics = self.metrics_generator.generate_performance_metrics()
            prompt = self.generate_prompt(metrics, word_count, test_id)
            response = await self.model.generate_content_async(prompt)
            content = response.text
           
            return {
                "email_number": email_number,
                "test_id": test_id,
                "content": content,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Failed to generate metrics report #{email_number}: {str(e)}")
            return {
                "email_number": email_number,
                "test_id": test_id,
                "content": None,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }


    async def generate_bulk_emails(self, word_count: int, num_emails: int,
                                 output_dir: str = "metric_reports") -> List[Dict]:
        self.logger.info(f"Starting bulk generation of {num_emails} metric reports")
        os.makedirs(output_dir, exist_ok=True)
        results = []
       
        batch_id = f"METRICS-BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}"
       
        for i in range(1, num_emails + 1):
            self.logger.info(f"Generating metric report {i}/{num_emails}")
           
            result = await self.generate_single_email(word_count, i)
            results.append(result)
           
            if result["status"] == "success":
                filename = f"metric_report_{result['test_id']}.txt"
                file_path = os.path.join(output_dir, filename)
               
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result["content"])
               
                # Save metrics data separately for analysis
                metrics_filename = f"metrics_{result['test_id']}.json"
                metrics_path = os.path.join(output_dir, metrics_filename)
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(result["metrics"], f, indent=2)
               
                self.logger.info(f"Saved report to {file_path}")
           
            if i < num_emails:
                time.sleep(EmailConfig.rate_limit_delay)
       
        # Generate enhanced batch report
        report = {
            "batch_id": batch_id,
            "generation_time": datetime.now().isoformat(),
            "configuration": {
                "word_count": word_count,
                "num_emails": num_emails,
                "output_directory": output_dir
            },
            "statistics": {
                "successful_generations": sum(1 for r in results if r["status"] == "success"),
                "failed_generations": sum(1 for r in results if r["status"] == "failed"),
                "average_word_count": sum(len(r["content"].split()) if r["content"] else 0
                                        for r in results) / len(results)
            },
            "metric_summaries": {
                "average_metrics": self._calculate_average_metrics(results),
                "metric_trends": self._analyze_metric_trends(results)
            },
            "results": results
        }
       
        report_path = os.path.join(output_dir, f"generation_report_{batch_id}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
       
        self.logger.info(f"Generation complete. Report saved to {report_path}")
        return results


    def _calculate_average_metrics(self, results: List[Dict]) -> Dict:
        """Calculate average metrics across all successful generations"""
        successful_metrics = [r["metrics"] for r in results if r["status"] == "success"]
        if not successful_metrics:
            return {}
       
        avg_metrics = {}
        for metric in successful_metrics[0].keys():
            values = [m[metric]["current"] for m in successful_metrics]
            avg_metrics[metric] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        return avg_metrics


    def _analyze_metric_trends(self, results: List[Dict]) -> Dict:
        """Analyze trends across all metrics in the batch"""
        successful_metrics = [r["metrics"] for r in results if r["status"] == "success"]
        if not successful_metrics:
            return {}
       
        trends = {}
        for metric in successful_metrics[0].keys():
            trends[metric] = {
                "improving": sum(1 for m in successful_metrics
                               if m[metric]["current"] > m[metric]["target"]),
                "declining": sum(1 for m in successful_metrics
                               if m[metric]["current"] < m[metric]["target"]),
                "stable": sum(1 for m in successful_metrics
                            if abs(m[metric]["current"] - m[metric]["target"]) <=
                            m[metric]["target"] * 0.05)
            }
        return trends


def main():
    parser = argparse.ArgumentParser(description="Generate smart metric test emails using Gemini AI")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--word-count", type=int, default=300,
                      help="Word count limit for each report")
    parser.add_argument("--num-emails", type=int, required=True,
                      help="Number of metric reports to generate")
    parser.add_argument("--output-dir", default="metric_reports",
                      help="Output directory for reports and metrics data")
    parser.add_argument("--min-trend-days", type=int, default=30,
                      help="Minimum number of days for trend analysis")
    parser.add_argument("--metrics-focus", choices=['performance', 'reliability', 'resource', 'all'],
                      default='all', help="Focus area for metrics generation")
    parser.add_argument("--urgency-level", choices=['low', 'medium', 'high'],
                      default='medium', help="Urgency level for metric anomalies")
    parser.add_argument("--include-recommendations", action='store_true',
                      help="Include AI-generated improvement recommendations")
    parser.add_argument("--verbose", action='store_true',
                      help="Enable detailed logging output")
   
    args = parser.parse_args()
   
    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
   
    # Validate word count
    if not (EmailConfig.min_word_count <= args.word_count <= EmailConfig.max_word_count):
        print(f"Error: Word count must be between {EmailConfig.min_word_count} and "
              f"{EmailConfig.max_word_count}")
        return
       
    # Validate number of emails
    if not (EmailConfig.min_emails <= args.num_emails <= EmailConfig.max_emails):
        print(f"Error: Number of emails must be between {EmailConfig.min_emails} and "
              f"{EmailConfig.max_emails}")
        return
       
    # Validate trend days
    if args.min_trend_days < 7:
        print("Error: Minimum trend days must be at least 7")
        return
   
    print("\n=== Smart Metric Report Generator ===")
    print(f"Configuration:")
    print(f"- Output Directory: {args.output_dir}")
    print(f"- Reports to Generate: {args.num_emails}")
    print(f"- Words per Report: {args.word_count}")
    print(f"- Metrics Focus: {args.metrics_focus}")
    print(f"- Urgency Level: {args.urgency_level}")
    print(f"- Trend Analysis: {args.min_trend_days} days")
    print("===================================")
   
    try:
        # Initialize generator with configuration
        generator = SmartTestEmailGenerator(args.api_key)
       
        # Configure metrics based on focus area
        if args.metrics_focus != 'all':
            generator.metrics_generator.set_focus(args.metrics_focus)
           
        # Configure urgency thresholds
        urgency_thresholds = {
            'low': 0.15,
            'medium': 0.10,
            'high': 0.05
        }
        generator.metrics_generator.set_threshold(urgency_thresholds[args.urgency_level])
       
        # Run the generation
        import asyncio
        results = asyncio.run(generator.generate_bulk_emails(
            args.word_count,
            args.num_emails,
            args.output_dir
        ))
       
        # Generate summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
       
        print("\n=== Generation Summary ===")
        print(f"Successfully generated: {successful}/{args.num_emails} reports")
        print(f"Failed generations: {failed}")
        print(f"Output directory: {args.output_dir}")
       
        # Calculate average metrics if any successful generations
        if successful > 0:
            print("\nMetrics Summary:")
            metrics_data = [r["metrics"] for r in results if r["status"] == "success"]
            for metric in metrics_data[0].keys():
                values = [m[metric]["current"] for m in metrics_data]
                avg = sum(values) / len(values)
                print(f"- Average {metric}: {avg:.2f}")
       
        print("\nDetailed reports and metrics data have been saved to the output directory.")
        print("=======================")
       
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during report generation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# python main.py --api-key AIzaSyDV4c47DjkrLYw7aCGSpao3L3Wkykw2JDk --num-emails 5 --word-count 150