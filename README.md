<p align="center">
  <img src="https://robo-bench.github.io/static/images/log/R1.png" alt="RoboBench Logo" width="120"/>
</p>

<h1 align="center" style="font-size:2.5em;">RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.17801v1)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://robo-bench.github.io/)
[![Huggingface](https://img.shields.io/badge/Huggingface-Repository-yellow)](https://huggingface.co/datasets/LeoFan01/RoboBench)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Module Documentation](#module-documentation)
  - [API Call General2](#api-call-general2)
  - [Multi-Choice Evaluation](#multi-choice-evaluation)
  - [Planning Evaluation](#planning-evaluation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 📋 Overview

RoboBench is a comprehensive evaluation benchmark designed to assess the capabilities of Multimodal Large Language Models (MLLMs) in embodied intelligence tasks. This benchmark provides a systematic framework for evaluating how well these models can understand and reason about robotic scenarios.

It provides comprehensive evaluation capabilities across three main dimensions:

- **🔄 API Call General2**: Handles multimodal API calls with image processing and prompt management
- **📊 Multi-Choice Evaluation**: Evaluates multiple-choice questions with automated scoring
- **🧠 Planning Evaluation**: Assesses complex planning tasks with DAG-based evaluation

## 🎯 Key Features

- **🧠 Comprehensive Benchmark**: Covers multiple aspects of embodied intelligence
- **📊 Rich Dataset**: Contains thousands of carefully curated examples
- **🔬 Novel Evaluation**: Designed with novel evaluation metrics
- **🌐 Multimodal**: Supports text, images, and video data
- **🤖 Robotics Focus**: Specifically tailored for robotic applications

---

## 🚀 Quick Start

### 1. Run Demo Scripts
For multi-choice questions, including Perception Reasoning, Affordance Reasoning, and Error Analysis:
```bash
cd evaluation/multi-choice
bash generation_pipeline_all.sh
```

For planning questions, including Instruction Comprehension and Generalized Planning:
```bash
cd evaluation/planning
bash generation_pipeline_all.sh
```

### 2. Basic API Call

```python
from evaluation.api_call_general2.general_pipeline import main
import asyncio

# Run basic evaluation
asyncio.run(main())
```

### 3. Multi-Choice Evaluation

```bash
cd evaluation/multi-choice
python evaluate_responses.py \
    --results_file results.json \
    --output evaluation_results.json \
    --openai-api-key "your-api-key"
```

### 4. Planning Evaluation

```bash
cd evaluation/planning
python all_in_one.py \
    --input-dir /path/to/results \
    --dataset-base-dir /path/to/RoboBench-dataset \
    --no-cache
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RoboBench
   ```

2. **Install dependencies**
   ```bash
   pip install openai aiofiles tenacity tqdm asyncio requests opencv-python
   ```

3. **Configure API credentials**
   ```python
   # Update api_utils.py with your OpenAI API key
   api_key = "your-openai-api-key-here"
   ```

---

## 📚 Module Documentation

### 🔄 API Call General2

The core infrastructure for handling multimodal API calls with advanced features.

#### Key Components

- **`api_utils.py`**: Asynchronous API utilities with retry logic and concurrency control
- **`cv_utils.py`**: Computer vision utilities for image processing and base64 encoding
- **`general_pipeline.py`**: Main processing pipeline with comprehensive argument parsing
- **`prompt_utils.py`**: Prompt management and formatting utilities

#### Features

- 🚀 **High Concurrency**: Up to 50 concurrent API calls
- 🔄 **Retry Logic**: Exponential backoff with configurable retry attempts (10 attempts)
- 💾 **Temporary Results**: Automatic saving of intermediate results
- 🖼️ **Image Processing**: Base64 encoding and image resizing (512x512)
- 📝 **Prompt Templates**: Flexible prompt formatting system

#### Usage Example

```python
from evaluation.api_call_general2.general_pipeline import main
import argparse

# Configure arguments
args = argparse.Namespace(
    questions_file="questions.json",
    output_file="output.jsonl",
    model="gpt-4o",
    image_key="image_urls",
    question_key="question"
)

# Run pipeline
main()
```

### 📊 Multi-Choice Evaluation

Comprehensive evaluation system for multiple-choice questions with automated scoring.

#### Key Components

- **`evaluate_responses.py`**: Core evaluation logic with GPT-based answer normalization
- **`generation_pipeline_all.sh`**: Batch processing script for multiple models
- **`merge_all_results.py`**: Result aggregation and merging utilities

#### Features

- 🎯 **Automated Scoring**: GPT-based answer extraction and normalization
- 📈 **Accuracy Calculation**: Comprehensive accuracy metrics
- 🔄 **Batch Processing**: Efficient processing of large datasets
- 📊 **Detailed Reports**: Rich evaluation reports with explanations

#### Usage Example

```python
from evaluation.multi_choice.evaluate_responses import evaluate_results_file

# Evaluate responses
evaluation_summary = evaluate_results_file(
    file_path="results.json",
    openai_api_key="your-api-key"
)

# Save results
save_evaluation_results(evaluation_summary, "evaluation_output.json")
```

### 🧠 Planning Evaluation

Advanced evaluation system for complex planning tasks with DAG-based assessment.

#### Key Components

- **`all_in_one.py`**: Comprehensive planning evaluator with multiple evaluation modes
- **`unified_prompts.py`**: Standardized prompt templates for consistent evaluation
- **`generation_pipeline_all.sh`**: Batch processing for planning tasks
- **`merge_all_results.py`**: Result aggregation utilities

#### Task Types

- **Q1 (Multi-step Planning)**: Complex planning tasks requiring multiple sequential actions
- **Q2 (Single-step Planning)**: Simple planning tasks requiring one action
- **Q3 (Yes/No Questions)**: Binary decision tasks

#### Features

- 🎯 **Multi-Modal Evaluation**: Q1, Q2, Q3 task types with specialized evaluation
- 📊 **DAG-Based Assessment**: Directed Acyclic Graph evaluation for complex tasks
- 🔄 **Caching System**: Intermediate result caching for efficient processing
- 📈 **Comprehensive Scoring**: Multiple scoring dimensions and metrics
- 🎨 **Visual Analysis**: Image-based constraint analysis

#### Usage Example

```python
from evaluation.planning.all_in_one import main
import asyncio

# Run comprehensive planning evaluation
result = asyncio.run(main(
    use_cache=True,
    skip_steps=[],
    input_dir="/path/to/results",
    dataset_base_dir="/path/to/dataset"
))
```

---

## 💡 Usage Examples

### Example 1: Basic Multimodal Evaluation

```bash
# Run evaluation with image processing
python evaluation/api_call_general2/general_pipeline.py \
    --questions_file data/questions.json \
    --output_file results/output.jsonl \
    --model gpt-4o \
    --image_key image_urls \
    --question_key question
```

### Example 2: Batch Multi-Choice Evaluation

```bash
# Run batch evaluation for multiple models
cd evaluation/multi-choice
bash generation_pipeline_all.sh
```

### Example 3: Comprehensive Planning Evaluation

```bash
# Run full planning evaluation pipeline
python evaluation/planning/all_in_one.py \
    --input-dir results/ \
    --dataset-base-dir /path/to/RoboBench-hf \
    --no-cache \
    --version v1.0
```

### Example 4: Skip Specific Steps

```bash
# Skip certain evaluation steps
python evaluation/planning/all_in_one.py \
    --input-dir results/ \
    --dataset-base-dir /path/to/RoboBench-hf \
    --skip-steps q1_extract q1_dag \
    --version v1.0
```

---

## ⚙️ Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
export BASE_DIR="/path/to/RoboBench-hf"
export RESULTS_DIR="/path/to/results"
```

### Model Configuration

```python
# Configure models and parameters
models = ["gpt-4o", "gpt-4", "claude-3"]
max_concurrent = 50
timeout = 360
retry_attempts = 10
```

### Evaluation Parameters

```python
# Planning evaluation parameters
evaluation_config = {
    "use_cache": True,
    "skip_steps": [],
    "dataset_base_dir": "/path/to/dataset",
    "version": "v1.0"
}
```

---

## 🔧 Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```python
   # Reduce concurrency
   MAX_CONCURRENT = 20
   ```

2. **Memory Issues**
   ```python
   # Process in smaller batches
   batch_size = 100
   ```

3. **Timeout Errors**
   ```python
   # Increase timeout
   TASK_TIMEOUT = 1200
   ```

4. **Missing DAG Files**
   ```bash
   # Ensure dataset base directory is correct
   --dataset-base-dir /path/to/RoboBench-hf
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Concurrency** | Up to 50 concurrent requests |
| **Retry Logic** | 10 attempts with exponential backoff |
| **Timeout** | 360 seconds per request |
| **Cache Support** | Intermediate result caching |
| **Batch Processing** | Efficient large-scale evaluation |
| **Image Processing** | 512x512 resize with base64 encoding |

---

## 🏗️ Architecture

```
evaluation/
├── api_call_general2/          # Core API calling infrastructure
│   ├── api_utils.py            # Async API utilities with retry logic
│   ├── cv_utils.py             # Computer vision utilities
│   ├── general_pipeline.py     # Main processing pipeline
│   └── prompt_utils.py         # Prompt management and formatting
├── multi-choice/               # Multiple-choice evaluation
│   ├── evaluate_responses.py   # Response evaluation logic
│   ├── generation_pipeline_all.sh  # Batch processing script
│   └── merge_all_results.py    # Result merging utilities
└── planning/                   # Planning task evaluation
    ├── all_in_one.py           # Comprehensive planning evaluator
    ├── generation_pipeline_all.sh  # Planning batch processing
    ├── merge_all_results.py    # Result aggregation
    └── unified_prompts.py      # Standardized prompt templates
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- The RoboBench team for the comprehensive evaluation framework
- Contributors and the open-source community

---

<div align="center">

**RoboBench Metric Evaluation Framework** - *Empowering the future of robotic AI evaluation*

[🔗 Documentation](https://github.com/your-repo/docs) | [🐛 Report Issues](https://github.com/your-repo/issues) | [💬 Discussions](https://github.com/your-repo/discussions)

</div>