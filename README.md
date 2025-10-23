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
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Framework](#evaluation-framework)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 📋 Overview

RoboBench is a comprehensive evaluation benchmark designed to assess the capabilities of Multimodal Large Language Models (MLLMs) in embodied intelligence tasks. This benchmark provides a systematic framework for evaluating how well these models can understand and reason about robotic scenarios.

The benchmark covers multiple dimensions of embodied intelligence:

- **📋 Instruction Comprehension**: Following and understanding complex instructions
- **🧠 Perception Reasoning**: Visual understanding and spatial reasoning
- **📊 Generalized Planning**: Multi-step task planning and execution
- **🎯 Affordance Reasoning**: Understanding object affordances and interactions
- **⚠️ Error Analysis**: Identifying and analyzing robotic errors

## 🎯 Key Features

- **🧠 Comprehensive Benchmark**: Covers multiple aspects of embodied intelligence
- **📊 Rich Dataset**: Contains thousands of carefully curated examples
- **🔬 Novel Evaluation**: Designed with novel evaluation metrics
- **🌐 Multimodal**: Supports text, images, and video data
- **🤖 Robotics Focus**: Specifically tailored for robotic applications
- **⚡ Automated Pipeline**: Complete evaluation pipeline with batch processing
- **📈 Detailed Analytics**: Comprehensive scoring and analysis tools

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

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- RoboBench dataset (download from [Hugging Face](https://huggingface.co/datasets/LeoFan01/RoboBench))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/lyl750697268/RoboBench
   cd RoboBench
   ```

2. **Install dependencies**
   ```bash
   conda create -n robobench python=3.10
   conda activate robobench
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   pip install huggingface_hub
   huggingface-cli login  
   huggingface-cli download LeoFan01/RoboBench --local-dir ./RoboBench_dataset
   ```

---

## 🚀 Quick Start
### 1. Configure API credentials
Complete your individual settings in:
```bash
`evaluation/multi-choice/generation_pipeline_all.sh`
`evaluation/planning/generation_pipeline_all.sh`
```

```bash
models=("gpt-4o") # models you want to eval

BASE_DIR="Your own path to the RoboBench-hf dataset"

OPENAI_API_KEY="Your own OpenAI API key"

BASE_URL="Your own base URL for the API"

RESULTS_DIR="Your own path to the results directory"

TARGET_DIRS=("1_instruction_comprehension" "2_perception_reasoning" "3_generalized_planning" "4_affordance_reasoning" "5_error_analysis") # dimensions you want to eval
```

### 2. Run Complete Evaluation Pipeline

**For Multi-Choice Questions** (Perception Reasoning, Affordance Reasoning, Error Analysis):
```bash
cd evaluation/multi-choice
bash generation_pipeline_all.sh
```

**For Planning Questions** (Instruction Comprehension, Generalized Planning):
```bash
cd evaluation/planning
bash generation_pipeline_all.sh
```

---

## 📁 Evaluation Framework

### 🔄 API Call General2

The core infrastructure for handling multimodal API calls with advanced features.

#### Key Components

- **`api_utils.py`**: Asynchronous API utilities with retry logic and concurrency control
- **`cv_utils.py`**: Computer vision utilities for image processing and base64 encoding
- **`general_pipeline.py`**: Main processing pipeline with comprehensive argument parsing
- **`prompt_utils.py`**: Prompt management and formatting utilities

#### Usage Example

```python
from evaluation.api_call_general2.general_pipeline import main
import argparse

# Configure arguments
args = argparse.Namespace(
    base_url="Your own base URL for the API",
    api_key="Your own OpenAI API key",
    questions_file="questions.json",
    output_file="output.jsonl",
    result_dir="/path/to/results",
    system_prompt_file="/path/to/RoboBench-dataset/system_prompt.json",
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

#### Usage Example

```python
from evaluation.multi_choice.evaluate_responses import evaluate_results_file

# Evaluate responses
evaluation_summary = evaluate_results_file(
    file_path="results.json",
    openai_api_key="Your own OpenAI API key",
    base_url="Your own base URL for the API"
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

#### Usage Example

```python
from evaluation.planning.all_in_one import main
import asyncio

# Run comprehensive planning evaluation
result = asyncio.run(main(
    use_cache=False,
    skip_steps=[], # type of tasks you don't want eval
    input_dir="/path/to/results",
    dataset_base_dir="/path/to/RoboBench-dataset",
    api_key="Your own OpenAI API key",
    base_url="Your own base URL for the API"
))
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

This project is licensed under the Apache2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Citation

```
@misc{luo2025robobenchcomprehensiveevaluationbenchmark,
      title={Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain}, 
      author={Yulin Luo and Chun-Kai Fan and Menghang Dong and Jiayu Shi and Mengdi Zhao and Bo-Wen Zhang and Cheng Chi and Jiaming Liu and Gaole Dai and Rongyu Zhang and Ruichuan An and Kun Wu and Zhengping Che and Shaoxuan Xie and Guocai Yao and Zhongxia Zhao and Pengwei Wang and Guang Liu and Zhongyuan Wang and Tiejun Huang and Shanghang Zhang},
      year={2025},
      eprint={2510.17801},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.17801}, 
}
```

---

<div align="center">

**RoboBench Metric Evaluation Framework** - *Empowering the future of robotic AI evaluation*

[🔗 Documentation](https://github.com/lyl750697268/RoboBench/blob/main/README.md) | [🐛 Report Issues](https://github.com/lyl750697268/RoboBench/issues) | [💬 Pull requests](https://github.com/lyl750697268/RoboBench/pulls)

</div>