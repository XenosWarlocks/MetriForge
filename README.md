# 🎯 MetriForge: Smart Test Email Generator for System Metrics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Generate intelligent, realistic test emails containing system performance metrics, trend analysis, and actionable insights using AI.

## 🚀 Features

- **Smart Metric Generation**: Creates realistic system performance, reliability, and resource utilization metrics
- **Trend Analysis**: Generates meaningful time-series data with configurable trends and volatility
- **AI-Powered Content**: Uses Google's Gemini AI to create contextually relevant email content
- **Customizable Focus Areas**: Target specific metric categories (performance, reliability, resource usage)
- **Professional Formatting**: ASCII tables, trend visualizations, and clear sectioning
- **Batch Processing**: Generate multiple unique test emails with consistent metrics
- **Detailed Reporting**: Comprehensive generation reports with metric summaries and trends

## 📋 Prerequisites

- Python 3.8 or higher
- Google Cloud API key with Gemini AI access
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/XenosWarlocks/metriforge.git
cd metriforge
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: source venv/Scripts/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💫 Usage

Basic usage with default settings:
```bash
python metriforge.py --api-key YOUR_API_KEY --num-emails 5
```

Advanced usage with customization:
```bash
python metriforge.py \
    --api-key YOUR_API_KEY \
    --num-emails 10 \
    --word-count 500 \
    --metrics-focus performance \
    --urgency-level high \
    --min-trend-days 60 \
    --include-recommendations \
    --verbose
```

## 🎨 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | Google Gemini AI API key | Required |
| `--num-emails` | Number of emails to generate | Required |
| `--word-count` | Words per email | 300 |
| `--output-dir` | Output directory | "metric_reports" |
| `--metrics-focus` | Focus area (performance/reliability/resource/all) | "all" |
| `--urgency-level` | Anomaly frequency (low/medium/high) | "medium" |
| `--min-trend-days` | Days of trend data | 30 |
| `--include-recommendations` | Add AI recommendations | False |
| `--verbose` | Detailed logging | False |

## 📊 Output Format

The generator creates three types of files for each run:

1. **Test Emails** (`test_email_*.txt`):
   - Professional format with headers
   - Executive summary
   - Metric tables and trends
   - Analysis and recommendations

2. **Metric Data** (`metrics_*.json`):
   - Raw metric values
   - Trend data
   - Targets and thresholds
   - Performance indicators

3. **Batch Report** (`generation_report_*.json`):
   - Generation statistics
   - Average metrics
   - Trend analysis
   - Success/failure rates

## 🌟 Example Output

```
+==================================================+
|            AUTOMATED METRICS REPORT               |
|            THIS IS A TEST EMAIL                   |
+==================================================+

Executive Summary:
Critical performance metrics show a 15% improvement in response time...

Current Metrics Performance:
+-----------------+----------+---------+----------+-----------+
| Metric          | Current  | Target  | Unit     | Status    |
+-----------------+----------+---------+----------+-----------+
| response_time   |    85.20 |   100.0 | ms       | NORMAL    |
| throughput      |  1025.50 |  1000.0 | req/s    | HIGH      |
+-----------------+----------+---------+----------+-----------+
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- Google Gemini AI for natural language generation
- The Python community for excellent libraries
- Contributors and testers

## 🔮 Future Plans

- [ ] Add support for custom metric templates
- [ ] Implement more sophisticated trend algorithms
- [ ] Add export to various formats (PDF, HTML)
- [ ] Create a web interface
- [ ] Add support for more AI models
- [ ] Implement real-time metric simulation

## 📧 Contact

For questions and support, please open an issue or contact the maintainers.

---

Made with ❤️ by [Xenos Warlocks]

