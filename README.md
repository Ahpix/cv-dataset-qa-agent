# CV Dataset QA Agent

AI Agent for automated Computer Vision dataset quality inspection using **Google ADK**, **Gemini 2.5 Flash**, **OpenCV**, and **Human-in-the-Loop (HITL)** approval.

---

# Overview

High-quality datasets are essential for building reliable computer vision models. Before model training, datasets should be inspected for issues such as corrupted images, invalid files, class imbalance, inconsistent image resolution, and overall dataset quality.

CV Dataset QA Agent automates this process using **Google Agent Development Kit (ADK)**. The agent analyzes a dataset, summarizes its statistics, optionally requests human approval through a Human-in-the-Loop step, and generates a professional Markdown quality report using Gemini.

This project was developed as the capstone project for the **Kaggle 5-Day AI Agents: Intensive Vibe Coding Course with Google**.

---

# Features

* Automatic dataset metadata extraction
* Image counting
* Average image resolution analysis
* Class distribution analysis
* Vision quality scoring
* Dataset quality inspection
* Human-in-the-Loop (HITL) approval
* Gemini-powered quality report generation
* Markdown report generation

---

# Architecture

```text
Dataset Folder
        │
        ▼
Parse Dataset Metadata
        │
        ▼
Vision Inspection
(OpenCV + Dataset Analysis)
        │
        ▼
Dataset Metrics
        │
        ▼
Human Approval (HITL)
        │
        ▼
Gemini Advisor
        │
        ▼
Markdown Quality Report
```

---

# Workflow

The workflow is implemented with **Google ADK**.

### 1. parse_dataset_metadata

* Reads the dataset directory
* Counts images
* Computes average image resolution
* Builds class distribution statistics

↓

### 2. vision_inspect_node

* Performs dataset quality inspection
* Computes overall vision quality score
* Collects detected quality issues

↓

### 3. prepare_advisor_input

* Converts inspection results into a structured JSON payload
* Passes the payload to the LLM agent

↓

### 4. hitl_gate

Requests Human-in-the-Loop approval before generating the final report.

↓

### 5. advisor_agent

Gemini 2.5 Flash generates a professional Markdown dataset quality report.

---

# Example Input

Run the ADK Web application.

```bash
adk web app
```

Then submit the following JSON:

```json
{
  "dataset_path": "F:\\Datasets\\demo_dataset",
  "dataset_metadata": {
    "dataset_name": "Intel Demo Dataset"
  }
}
```

---

# Example Output

```markdown
# Vision Dataset QA Report

## Dataset Overview

Dataset Name: demo_dataset

Total Images: 300

Average Resolution: 150 × 149

Vision Quality Score: 1.00

---

## Class Distribution

- buildings : 50 images
- forest : 50 images
- glacier : 50 images
- mountain : 50 images
- sea : 50 images
- street : 50 images

---

## Label Issues

No issues detected.

---

## Recommendations

- Dataset quality is excellent.
- The dataset is suitable for computer vision classification tasks.
- Conduct routine quality inspection before future training.
```

---

# Demo Dataset

A lightweight demonstration dataset is included in this repository.

```text
demo_dataset/

├── buildings
├── forest
├── glacier
├── mountain
├── sea
└── street
```

The repository also includes

```text
make_demo_dataset.py
```

which creates the demo dataset from the Intel Image Classification dataset.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Ahpix/cv-dataset-qa-agent.git

cd cv-dataset-qa-agent
```

Install dependencies.

```bash
uv sync
```

Launch the ADK Web interface.

```bash
adk web app
```

---

# Repository Structure

```text
app/
    agent.py

demo_dataset/

make_demo_dataset.py

schemas.py

tools.py

pyproject.toml

uv.lock
```

---

# Technology Stack

* Google ADK 2.3
* Gemini 2.5 Flash
* Python
* OpenCV
* Pillow
* NumPy
* Pydantic

---

# Future Improvements

* COCO dataset support
* Pascal VOC dataset support
* YOLO dataset support
* Object detection dataset inspection
* Segmentation dataset inspection
* Interactive dashboard
* Automatic visualization of dataset statistics

---

# Acknowledgements

This project was created as the capstone project for the **Kaggle 5-Day AI Agents: Intensive Vibe Coding Course with Google**.

It demonstrates the use of Google ADK workflows, Gemini-powered reasoning, Human-in-the-Loop approval, and computer vision dataset inspection for practical AI agent development.
