<p align="center">
  <img src="assets/R1.png" alt="RoboBench Logo" width="120"/>
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
- [Results](#results)
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
evaluation/multi-choice/generation_pipeline_all.sh
evaluation/planning/generation_pipeline_all.sh
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

## 😲 Results
<h1>Results on Perception Reasoning (%)</h1>
<p><strong>Attr.</strong> = Attribute; <strong>Temp.</strong> = Temporal; <strong>Refer. Comprehen.</strong> = Reference Comprehension</p>
<table>
<tr style="background:#f2f2f2;"><th></th><th colspan="9"><strong>Perception Reasoning</strong></th></tr>
<tr><th rowspan="2"><strong>Model</strong></th><th colspan="2"><strong>Robotic-centric</strong></th><th colspan="2"><strong>Object-centric</strong></th><th colspan="3"><strong>Scene-centric</strong></th><th><strong>Task-centric</strong></th><th rowspan="2"><strong>Avg</strong></th></tr>
<tr><th>Robot-type</th><th>Robot-view</th><th>Static Attr.</th><th>Functional Attr.</th><th>Spatial Relation</th><th>Temp. Grounding</th><th>Causality</th><th>Refer. Comprehen.</th></tr>
<tr style="background:#f2f2f2;"><td colspan="10"><strong>Basic Reference</strong></td></tr>
<tr><td>Human Evaluation</td><td>80.67</td><td>79.08</td><td>43.77</td><td>83.89</td><td>70.91</td><td>51.61</td><td>91.22</td><td>93.22</td><td>74.3</td></tr>

<tr><td>GPT-4o-text-only</td><td>20.51</td><td>13.77</td><td>5.18</td><td>35.37</td><td>25.74</td><td>18.32</td><td>25.52</td><td>22.09</td><td>20.81</td></tr>

<tr style="background:#f2f2f2;"><td colspan="10"><strong>Closed-Source MLLMs</strong></td></tr>
<tr><td>GPT-4o-Mini</td><td>38.75</td><td>18.84</td><td>26.43</td><td>53.66</td><td>30.36</td><td>22.65</td><td>34.25</td><td>39.67</td><td>33.08</td></tr>

<tr><td>GPT-4o</td><td><b>64.96</b></td><td>39.38</td><td>24.92</td><td>46.75</td><td>42.24</td><td>20.61</td><td>33.1</td><td>41.31</td><td>39.16</td></tr>

<tr><td>Claude-3.5-Sonnet</td><td>41.31</td><td>36.23</td><td>29.13</td><td>62.6</td><td>34.98</td><td>21.88</td><td>36.09</td><td>25.36</td><td>35.95</td></tr>

<tr><td>Claude-3.7-Sonnet</td><td>40.46</td><td>32.37</td><td>45.2</td><td>71.14</td><td>36.63</td><td>21.09</td><td>40.92</td><td>28.02</td><td>39.48</td></tr>

<tr><td>Gemini-2.0-Flash</td><td>56.69</td><td>20.77</td><td>49.08</td><td><u>78.46</u></td><td>42.57</td><td>21.37</td><td>51.72</td><td>72.4</td><td>49.13</td></tr>

<tr><td>Gemini-2.5-Flash</td><td>62.39</td><td>39.38</td><td><b>55.02</b></td><td>77.24</td><td><u>57.43</u></td><td><u>33.58</u></td><td><u>70.34</u></td><td><u>74.64</u></td><td><u>58.75</u></td></tr>

<tr><td>Gemini-2.5-Pro</td><td><u>64.30</u></td><td>41.71</td><td><u>54.83</u></td><td><b>82.27</b></td><td><b>60.44</b></td><td><b>49.68</b></td><td><b>71.73</b></td><td><b>78.68</b></td><td><b>62.96</b></td></tr>

<tr><td>Qwen-VL-Plus</td><td>28.21</td><td>21.74</td><td>34.63</td><td>58.54</td><td>27.72</td><td>21.37</td><td>31.03</td><td>34.36</td><td>32.2</td></tr>

<tr><td>Qwen-VL-Max</td><td>47.86</td><td><b>43.48</b></td><td>39.7</td><td>75.2</td><td>50.17</td><td>27.45</td><td>37.93</td><td>41.53</td><td>45.42</td></tr>

<tr style="background:#f2f2f2;"><td colspan="10"><strong>Open-Source Multi-Image MLLMs</strong></td></tr>
<tr><td>LLaVA-OneVision-0.5B</td><td>30.34</td><td>23.68</td><td>37.08</td><td>49.66</td><td>27.27</td><td>18.42</td><td>23.65</td><td>19.21</td><td>28.66</td></tr>

<tr><td>LLaVA-OneVision-7B</td><td>44.83</td><td>30.26</td><td>33.43</td><td>75.84</td><td>45.45</td><td>23.68</td><td>25.68</td><td>44.63</td><td>40.48</td></tr>

<tr><td>Qwen2.5-VL-7B-Ins</td><td>23.93</td><td>26.81</td><td>37.86</td><td>46.34</td><td>31.68</td><td>22.9</td><td>14.48</td><td>36.81</td><td>30.1</td></tr>

<tr><td>Qwen2.5-VL-72B-Ins</td><td>47.72</td><td><u>42.75</u></td><td>41.74</td><td>72.95</td><td>48.51</td><td>27.87</td><td>40.32</td><td>42.13</td><td>45.5</td></tr>

<tr style="background:#f2f2f2;"><td colspan="10"><strong>Embodied MLLMs</strong></td></tr>
<tr><td>RoboBrain-2.0-7B</td><td>44.97</td><td>24.84</td><td>40.43</td><td>79.19</td><td>48.18</td><td>23.48</td><td>41.22</td><td>53.67</td><td>44.5</td></tr>

</table>

<h1>Results on Instruction Comprehension and Generalized Planning Q1</h1>
<p><strong>manip.</strong> = manipulation; <strong>Afford.</strong> = affordance; <strong>Knowl.</strong> = Knowledge</p>
<table>
<tr style="background:#e6e6e6;"><th><strong>Model</strong></th><th colspan="3"><strong>Instruction Comprehension</strong></th><th colspan="11"><strong>Generalized Planning</strong></th></tr>
<tr><th></th><th><strong>Explicit</strong></th><th><strong>Implicit</strong></th><th><strong>Avg</strong></th><th colspan="4"><strong>Cross-Embodiment Planning</strong></th><th colspan="3"><strong>Cross-Object Planning</strong></th><th colspan="2"><strong>Cross-View Planning</strong></th><th><strong>Cross-Task Planning</strong></th><th><strong>Avg</strong></th></tr>
<tr><th></th><th></th><th></th><th></th><th>Single-arm</th><th>Dual-arm</th><th>Mobile-manip.</th><th>Human</th><th>Material Afford.</th><th>Physical Attr.</th><th>World Knowl.</th><th>Multi</th><th>Single</th><th>Navigation Plan</th><th>Avg</th></tr>
<tr style="background:#e6e6e6;"><td colspan="15"><strong>Basic Reference</strong></td></tr>
<tr><td>Human Evaluation</td><td>59.94</td><td>61.13</td><td>60.54</td><td>72.5</td><td>41.93</td><td>41.55</td><td>62.28</td><td>56.7</td><td>58.98</td><td>49.36</td><td>52.82</td><td>51.59</td><td>45.23</td><td>54.5</td></tr>

<tr><td>GPT-4o-text-only</td><td>38.8</td><td>11.1</td><td>24.95</td><td>26.7</td><td>33.32</td><td>43.65</td><td>37.86</td><td>36.58</td><td>22.33</td><td>37.68</td><td>44.35</td><td>38.11</td><td>36.9</td><td>33.95</td></tr>

<tr style="background:#e6e6e6;"><td colspan="15"><strong>Closed-Source MLLMs</strong></td></tr>
<tr><td>GPT-4o-Mini</td><td>41.21</td><td>14.95</td><td>28.08</td><td>27.47</td><td>25.21</td><td>37.98</td><td>31.72</td><td>33.75</td><td>38.46</td><td>42.56</td><td>39.11</td><td>33.29</td><td>34.04</td><td>33.31</td></tr>

<tr><td>GPT-4o</td><td>45.6</td><td><u>19.04</u></td><td>32.32</td><td>28.28</td><td>32.65</td><td><b>52.69</b></td><td>35.71</td><td>39.93</td><td>46.09</td><td>41.34</td><td>38.51</td><td>33.66</td><td>39.41</td><td>37.74</td></tr>

<tr><td>Claude-3.5-Sonnet</td><td>42.11</td><td>14.85</td><td>28.48</td><td><b>30.18</b></td><td>33.65</td><td>50.29</td><td><b>41.05</b></td><td>38.28</td><td>40.67</td><td>39.63</td><td>45.95</td><td>40.43</td><td>39.77</td><td>38.07</td></tr>

<tr><td>Claude-3.7-Sonnet</td><td><u>47.77</u></td><td>14.53</td><td>31.15</td><td><u>29.86</u></td><td><u>38.69</u></td><td>50.39</td><td>37.06</td><td>38.65</td><td>41.86</td><td><b>51.83</b></td><td><b>48.19</b></td><td><u>44.51</u></td><td><u>39.95</u></td><td><u>41.68</u></td></tr>

<tr><td>Gemini-2.0-Flash</td><td>43.49</td><td>16.38</td><td>29.93</td><td>28.67</td><td>33.66</td><td>48.27</td><td>33.95</td><td><b>40.76</b></td><td><u>54.27</u></td><td>40.12</td><td>46.13</td><td>40.73</td><td>37.02</td><td>38.62</td></tr>

<tr><td>Gemini-2.5-Flash</td><td>42.53</td><td>17.1</td><td>29.82</td><td>27.05</td><td><b>40.46</b></td><td>49.91</td><td>34.5</td><td>39.87</td><td>53.37</td><td><u>46.22</u></td><td>39.41</td><td>43.29</td><td>38.32</td><td>39.33</td></tr>

<tr><td>Gemini-2.5-Pro</td><td><b>51.15</b></td><td><b>19.60</b></td><td><b>35.37</b></td><td>29.71</td><td>37.65</td><td><u>50.96</u></td><td><u>37.44</u></td><td>39.29</td><td><b>56.50</b></td><td>43.29</td><td><u>47.35</u></td><td><b>45.12</b></td><td><b>43.62</b></td><td><b>41.81</b></td></tr>

<tr><td>Qwen-VL-Plus</td><td>37.77</td><td>10.38</td><td>24.07</td><td>24.68</td><td>21.75</td><td>32.98</td><td>33.91</td><td>28.45</td><td>33.55</td><td>33.78</td><td>30.95</td><td>28.6</td><td>4.39</td><td>26.77</td></tr>

<tr><td>Qwen-VL-Max</td><td>46.45</td><td>16.98</td><td><u>31.71</u></td><td>28.3</td><td>35.73</td><td>47.79</td><td>32.4</td><td><u>40.44</u></td><td>44.33</td><td>42.32</td><td>41.79</td><td>37.68</td><td>38.0</td><td>37.68</td></tr>

<tr style="background:#e6e6e6;"><td colspan="15"><strong>Open-Source Multi-Image MLLMs</strong></td></tr>
<tr><td>LLaVA-OneVision-0.5B</td><td>6.82</td><td>1.24</td><td>3.61</td><td>2.9</td><td>4.57</td><td>4.77</td><td>3.68</td><td>4.77</td><td>3.47</td><td>6.47</td><td>4.3</td><td>3.62</td><td>11.39</td><td>4.83</td></tr>

<tr><td>LLaVA-OneVision-7B</td><td>18.93</td><td>3.48</td><td>10.05</td><td>11.48</td><td>16.23</td><td>8.27</td><td>5.34</td><td>18.51</td><td>15.62</td><td>8.1</td><td>0.0</td><td>15.16</td><td>24.67</td><td>12.15</td></tr>

<tr><td>Qwen2.5-VL-7B-Ins</td><td>26.45</td><td>4.65</td><td>15.55</td><td>19.47</td><td>12.9</td><td>28.75</td><td>28.19</td><td>22.06</td><td>21.63</td><td>25.61</td><td>11.79</td><td>20.12</td><td>2.1</td><td>18.64</td></tr>

<tr><td>Qwen2.5-VL-72B-Ins</td><td>46.81</td><td>15.15</td><td>30.98</td><td>28.2</td><td>36.92</td><td>49.14</td><td>31.31</td><td>40.51</td><td>44.94</td><td>38.9</td><td>43.16</td><td>40.24</td><td>37.47</td><td>37.73</td></tr>

<tr style="background:#e6e6e6;"><td colspan="15"><strong>Embodied MLLMs</strong></td></tr>
<tr><td>RoboBrain-2.0-7B</td><td>36.93</td><td>8.19</td><td>22.51</td><td>15.46</td><td>25.32</td><td>32.72</td><td>31.81</td><td>19.85</td><td>30.85</td><td>23.24</td><td>31.51</td><td>23.89</td><td>24.53</td><td>25.35</td></tr>

</table>

<h1>Results on Affordance Prediction and Failure Analysis</h1>
<p><strong>Naviga.</strong> = Navigation</p>
<table>
<tr style="background:#e6e6e6;"><th><strong>Model</strong></th><th colspan="4"><strong>Affordance Prediction</strong></th><th colspan="3"><strong>Failure Analysis</strong></th></tr>
<tr><th></th><th>Static</th><th>Dynamic</th><th>Naviga.</th><th>Avg</th><th>Execution</th><th>Planning</th><th>Avg</th></tr>
<tr style="background:#e6e6e6;"><td colspan="8"><strong>Basic Reference</strong></td></tr>
<tr><td>Human Evaluation</td><td>86.08</td><td>80.02</td><td>81.85</td><td>82.63</td><td>47.3</td><td>80.67</td><td>63.99</td></tr>

<tr><td>GPT-4o-text-only</td><td>44.89</td><td>40.7</td><td>38.19</td><td>39.88</td><td>25.17</td><td>37.93</td><td>31.55</td></tr>

<tr style="background:#e6e6e6;"><td colspan="8"><strong>Closed-Source MLLMs</strong></td></tr>
<tr><td>GPT-4o-Mini</td><td>50.64</td><td>42.88</td><td>42.3</td><td>46.39</td><td>17.66</td><td>44.6</td><td>31.13</td></tr>

<tr><td>GPT-4o</td><td>55.61</td><td>49.14</td><td>49.91</td><td>51.91</td><td>22.29</td><td>57.01</td><td>39.65</td></tr>

<tr><td>Claude-3.5-Sonnet</td><td>56.26</td><td>54.25</td><td>53.84</td><td>54.77</td><td>16.12</td><td>47.52</td><td>31.82</td></tr>

<tr><td>Claude-3.7-Sonnet</td><td>60.02</td><td>52.38</td><td>50.07</td><td>54.06</td><td>18.32</td><td>54.24</td><td>36.28</td></tr>

<tr><td>Gemini-2.0-Flash</td><td>61.65</td><td><u>61.76</u></td><td><b>66.89</b></td><td><u>63.37</u></td><td><u>28.48</u></td><td>59.8</td><td><u>44.14</u></td></tr>

<tr><td>Gemini-2.5-Flash</td><td>61.2</td><td>52.04</td><td>52.01</td><td>54.29</td><td>18.54</td><td><u>67.65</u></td><td>43.1</td></tr>

<tr><td>Gemini-2.5-Pro</td><td><u>70.54</u></td><td><b>62.03</b></td><td><u>63.96</u></td><td><b>65.21</b></td><td>15.96</td><td><b>74.31</b></td><td><b>45.14</b></td></tr>

<tr><td>Qwen-VL-Plus</td><td>51.74</td><td>37.42</td><td>47.97</td><td>48.18</td><td>13.91</td><td>40.0</td><td>26.96</td></tr>

<tr><td>Qwen-VL-Max</td><td>70.01</td><td>56.26</td><td>50.85</td><td>59.43</td><td>17.22</td><td>57.93</td><td>37.58</td></tr>

<tr style="background:#e6e6e6;"><td colspan="8"><strong>Open-Source Multi-Image MLLMs</strong></td></tr>
<tr><td>LLaVA-OneVision-0.5B</td><td>20.56</td><td>28.56</td><td>27.69</td><td>24.76</td><td>21.19</td><td>24.67</td><td>22.93</td></tr>

<tr><td>LLaVA-OneVision-7B</td><td>23.83</td><td>33.61</td><td>33.43</td><td>30.29</td><td><b>29.14</b></td><td>34.0</td><td>31.56</td></tr>

<tr><td>Qwen2.5-VL-7B-Ins</td><td>49.73</td><td>38.03</td><td>42.16</td><td>43.15</td><td>13.91</td><td>26.9</td><td>20.41</td></tr>

<tr><td>Qwen2.5-VL-72B-Ins</td><td><b>71.54</b></td><td>51.94</td><td>47.67</td><td>56.67</td><td>12.59</td><td>50.72</td><td>31.66</td></tr>

<tr style="background:#e6e6e6;"><td colspan="8"><strong>Embodied MLLMs</strong></td></tr>
<tr><td>RoboBrain-2.0-7B</td><td>51.87</td><td>54.63</td><td>41.61</td><td>49.37</td><td>7.95</td><td>42.0</td><td>41.24</td></tr>

</table>

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