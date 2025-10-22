# MedNuggetizer - ECIR 2026 Demo

**Reproducibility-Supporting Tool for LLM-Based Medical Information Nugget Extraction**

This repository contains the complete source code for the medical information nugget extraction system demonstrated at ECIR 2026. It provides a web-based application and REST API for extracting, clustering, and ranking information nuggets from medical PDF documents using Large Language Models (LLMs).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [REST API](#rest-api)
  - [Docker Deployment](#docker-deployment)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Research & Citation](#research--citation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

MedNuggetizer is an LLM-powered system designed to extract and aggregate key information from medical literature. The system addresses the reproducibility challenge in LLM-based extraction by:

1. **Multiple Extraction Runs**: Running the extraction process multiple times on the same document
2. **Clustering & Filtering**: Grouping similar nuggets and retaining only those that appear consistently across runs
3. **Evidence Ranking**: Ranking clusters by size to indicate evidence strength across multiple documents

This approach ensures that only reproducible, consistently extracted information is presented to users.

---

## Key Features

- **Multi-Document Processing**: Upload and process multiple PDF files simultaneously
- **Query-Guided Extraction**: Query parameter to focus extraction on specific topics
- **Reproducibility Filtering**: Configurable confidence thresholds to ensure consistent results
- **Evidence Strength Ranking**: Automatic ranking based on cluster size and document coverage
- **Real-Time Progress Tracking**: Server-Sent Events (SSE) for live processing updates
- **Web UI & REST API**: Both interactive interface and programmatic access
- **Production-Ready**: Docker support with health checks and proper logging
- **CPU-Optimized**: Uses PyTorch CPU-only for faster deployment without GPU requirements

---

## System Architecture

The system consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                        Flask Web App                         │
│                    (app.py / demo.py)                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Nugget     │  │  Clustering  │  │ Summarizers  │
    │  Detection   │  │   (BERTopic) │  │   (Gemini)   │
    │  (Gemini)    │  │              │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### Workflow:

1. **PDF Upload**: User uploads PDF files via web UI or API
2. **Extraction**: LLM extracts information nuggets (n runs per PDF)
3. **Per-PDF Clustering**: BERTopic clusters nuggets within each PDF
4. **Filtering**: Keep only clusters with exactly n nuggets (one per run)
5. **Global Clustering**: Merge filtered nuggets across all PDFs and recluster
6. **Summarization**: Generate concise headings for each cluster
7. **Ranking**: Sort clusters by size (evidence strength)
8. **Display**: Present ranked results with confidence scores

---

## Requirements

### Software Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for multiple PDFs)
- **Disk Space**: ~2GB for dependencies and models

### API Keys

You need a Google Gemini API key. The system supports multiple LLM providers:
- **Google Gemini** (default, recommended): For nugget detection and summarization
- OpenAI GPT (optional)
- Anthropic Claude (optional)

Get your Gemini API key: https://ai.google.dev/

---

## Installation

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamyAteia/mednuggetizer-ecir2026.git
   cd mednuggetizer-ecir2026/application
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```

### Option 2: Docker Installation

```bash
# Build the Docker image
docker build -t mednuggetizer:latest .

# Or use docker-compose
docker-compose up -d
```

---

## Configuration

### API Key Setup

The application requires a Google Gemini API key. You can provide it in two ways:

#### Method 1: Environment Variable (Recommended)
```bash
export GEMINI_API_KEY="your-api-key-here"
```

On Windows:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

#### Method 2: Configuration File
Create a file named `google_api_key.txt` in the project root:
```bash
echo "your-api-key-here" > google_api_key.txt
```

### Docker Environment Variables

When using Docker, set environment variables in a `.env` file:
```bash
GEMINI_API_KEY=your-api-key-here
FLASK_ENV=production
```

Then reference it in `docker-compose.yml` (already configured).

### Application Configuration

Key configuration parameters in `app.py`:
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 50MB)
- `UPLOAD_FOLDER`: Temporary storage for uploads (default: `/tmp/uploads`)
- `PORT`: Application port (default: 4000)

---

## Usage

### Web Interface

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser**:
   Navigate to `http://localhost:4000`

3. **Upload PDFs**:
   - Click "Upload Files" and select one or more PDF documents
   - (Optional) Enter a query to guide extraction (e.g., "diabetes treatment guidelines")
   - Set number of runs (default: 3) - higher values increase reproducibility confidence
   - Set LLM confidence threshold (default: 0.8) - higher values require more consistent extraction

4. **View Results**:
   - Watch real-time progress updates
   - Explore ranked clusters with evidence strength
   - Review nuggets with confidence scores and source attributions

### REST API

#### Extract Nuggets Endpoint

**POST** `/api/extract`

**Request**:
```bash
curl -X POST http://localhost:4000/api/extract \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "query=What are the treatment options for type 2 diabetes?" \
  -F "n_runs=3" \
  -F "llm_confidence=0.8"
```

**Parameters**:
- `files` (required): One or more PDF files
- `query` (optional): Query string to guide extraction
- `n_runs` (optional): Number of extraction runs per PDF (1-10, default: 3)
- `llm_confidence` (optional): Confidence threshold (0.0-1.0, default: 0.8)
- `session_id` (optional): UUID for progress tracking

**Response**:
```json
{
  "status": "success",
  "n_pdfs": 2,
  "n_runs": 3,
  "total_nuggets": 24,
  "n_clusters": 5,
  "clusters": [
    {
      "cluster_id": 1,
      "cluster_heading": "Metformin as First-Line Therapy",
      "nuggets": [
        "Metformin is recommended as initial pharmacologic treatment... (Confidence: 100%)",
        "First-line therapy should be metformin unless contraindicated... (Confidence: 100%)"
      ],
      "sources": ["document1.pdf", "document2.pdf"],
      "size": 8
    }
  ],
  "pdf_names": ["document1.pdf", "document2.pdf"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Progress Tracking Endpoint

**GET** `/api/progress/<session_id>`

Server-Sent Events (SSE) stream for real-time progress updates.

**Example**:
```javascript
const eventSource = new EventSource('/api/progress/' + sessionId);
eventSource.onmessage = function(event) {
  const progress = JSON.parse(event.data);
  console.log(`Step ${progress.step}: ${progress.detail} (${progress.percentage}%)`);
};
```

#### Health Check Endpoint

**GET** `/api/health`

```json
{
  "status": "healthy",
  "service": "Medical Nugget Extraction API"
}
```

#### API Information Endpoint

**GET** `/api/info`

Returns comprehensive API documentation and usage instructions.

### Docker Deployment

#### Using Docker Compose (Recommended)

1. **Configure environment**:
   ```bash
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

2. **Start the service**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   Navigate to `http://localhost:8547`

4. **View logs**:
   ```bash
   docker-compose logs -f ecir-nugget-extractor
   ```

5. **Stop the service**:
   ```bash
   docker-compose down
   ```

#### Using Docker directly

```bash
# Build
docker build -t mednuggetizer:latest .

# Run
docker run -d \
  -p 4000:4000 \
  -e GEMINI_API_KEY="your-api-key-here" \
  -v uploads:/tmp/uploads \
  --name mednuggetizer \
  mednuggetizer:latest

# Check health
curl http://localhost:4000/api/health
```

---

## How It Works

### 1. Multiple Extraction Runs

The system runs LLM-based extraction multiple times (configurable `n_runs`, default: 3) on each PDF to account for LLM variability:

```
Run 1: ["diabetes diagnosis", "metformin treatment", "blood glucose 180"]
Run 2: ["type 2 diabetes confirmed", "metformin prescribed", "glucose elevated"]
Run 3: ["diabetes mellitus type 2", "metformin therapy", "high blood sugar"]
```

### 2. Per-PDF Clustering

Using BERTopic (sentence transformers + HDBSCAN), similar nuggets are grouped:

```
Cluster A (3 nuggets): diabetes-related ✓
Cluster B (3 nuggets): metformin-related ✓
Cluster C (3 nuggets): glucose-related ✓
Cluster D (1 nugget): follow-up ✗ (discarded)
```

### 3. Reproducibility Filtering

Only clusters with `min_cluster_size = n_runs * llm_confidence` are retained. With `n_runs=3` and `llm_confidence=0.8`, clusters need ≥2 nuggets (one from different runs).

### 4. Global Clustering & Ranking

Filtered nuggets from all PDFs are merged and reclustered:

```
Cluster 1 (12 nuggets from 4 PDFs): "Metformin as first-line treatment"
  → HIGH evidence strength
  
Cluster 2 (9 nuggets from 3 PDFs): "HbA1c monitoring importance"
  → MEDIUM-HIGH evidence strength
  
Cluster 3 (3 nuggets from 1 PDF): "Insulin therapy"
  → LOW evidence strength
```

### 5. Summarization

Each cluster receives an LLM-generated heading summarizing the key information.

---


## Research & Citation

This system implements a novel approach to reproducible information extraction from medical literature using LLMs. Key innovations:

1. **Multi-run extraction** to capture LLM variability
2. **Clustering-based filtering** to ensure reproducibility
3. **Evidence strength ranking** based on cross-document consistency

If you use this system in your research, please cite:

```bibtex
TODO
```

### Related Work

- BERTopic: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure
- Sentence Transformers: Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- HDBSCAN: McInnes et al. (2017). hdbscan: Hierarchical density based clustering

---

## Troubleshooting

### Debug Mode

Enable detailed logging:
```bash
export FLASK_DEBUG=1
python app.py
```

Check logs for detailed error traces:
```bash
# Docker logs
docker-compose logs -f ecir-nugget-extractor
```

---

## Acknowledgments

This work was developed for demonstration at ECIR 2026. We thank:
- The ECIR 2026 organizing committee
- Contributors to BERTopic, Sentence Transformers, and HDBSCAN
- Google for providing the Gemini API

---

