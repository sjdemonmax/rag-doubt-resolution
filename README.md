# RAG-Based Student Doubt Resolution Pipeline

> **Role**: Senior AI Engineer · Entrance-Exam EdTech Platform

## The Problem

In high-stakes entrance exam preparation, students generate hundreds of doubts per day across Physics, Chemistry, Mathematics, and Biology. The traditional model — routing every query to a human teacher — creates catastrophic bottlenecks: teachers are overwhelmed, response latency climbs to hours, and learner momentum stalls at precisely the moments that matter most. At the same time, deploying a fully automated AI system without safeguards risks a worse outcome: hallucinated, confidently wrong answers that actively harm student understanding.

The core challenge is knowing *when to trust the AI* and when to escalate to a human — without manually reviewing every query.

## The Solution

This project implements a **production-grade RAG (Retrieval-Augmented Generation) doubt resolution pipeline** with ChromaDB vector retrieval and a calibrated confidence-based human fallback mechanism.

### How It Works

1. **Dense Vector Retrieval** — Each incoming student query is encoded and matched against a curriculum knowledge base using cosine similarity to surface the top-3 most relevant context chunks.
2. **Confidence Scoring** — Retrieval confidence is derived from cosine distance, producing a continuous score that reflects how well the knowledge base covers the query.
3. **Automated Resolution** — High-confidence queries receive a grounded, context-anchored answer immediately with zero human involvement.
4. **Safe Escalation** — Low-confidence queries are automatically routed to the teacher queue, preventing hallucinated responses from reaching students.
5. **Threshold Calibration** — The confidence cutoff is empirically tuned using ROC-AUC and Precision-Recall analysis to maximise routing accuracy while minimising false positives.

### Tech Stack

- **Vector Store**: ChromaDB (local, privacy-preserving)
- **Retrieval**: Dense semantic embeddings via SentenceTransformers
- **Analysis**: Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
- **Dashboard**: Streamlit (Academic Operations Dashboard — live threshold sensitivity, escalation queue audit, resolution metrics)
- **Notebook**: JupyterLab (fully executed, cell-by-cell benchmarking)

## Measurable Business Impact

| Metric | Result |
|---|---|
| Routing accuracy | **> 85%** |
| Target auto-resolution rate | Majority of high-confidence queries resolved with zero human touch |
| False positive rate (hallucinations reaching students) | Minimised through calibrated threshold |
| Teacher queue reduction | Significant deflection of routine queries away from human educators |
| Response latency | Near-instantaneous for auto-resolved queries vs. hours for manual routing |

By calibrating the confidence threshold empirically rather than arbitrarily, the system eliminates the binary choice between *"trust AI fully"* and *"route everything to humans"* — replacing it with a principled, auditable, continuously tunable decision boundary that scales with the platform.

## Repository Contents

| File | Description |
|---|---|
|  | Starter notebook template with scaffolding |
|  | Fully executed benchmarking notebook |
|  | Synthetic dataset generation script |
|  /  | Streamlit Academic Operations Dashboard |
|  | Logged resolution outcomes per query |
|  | Benchmark query evaluation dataset |
|  | Curriculum knowledge base (chunked) |
