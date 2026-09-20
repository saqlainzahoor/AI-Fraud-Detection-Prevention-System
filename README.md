# AI-Based Digital Banking Fraud Detection & Prevention System

An end-to-end machine learning system that scores banking transactions in real-time, classifies them into risk levels, explains *why* a transaction looks suspicious, and recommends an action — built for the FinTech/Risk Management domain.

**Live Demo:** _[Streamlit link — add after deployment]_
**API Docs:** _[API link — add after deployment, if deployed]_

---

## Table of Contents
- [Business Problem](#business-problem)
- [System Flow](#system-flow)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Validation Strategy](#validation-strategy)
- [Baseline](#baseline)
- [Modeling & Model Comparison](#modeling--model-comparison)
- [A Critical Finding: The Overfitting Trap](#a-critical-finding-the-overfitting-trap)
- [Final Model & Threshold Analysis](#final-model--threshold-analysis)
- [Risk Scoring](#risk-scoring)
- [Explainability](#explainability)
- [Dashboard](#dashboard)
- [API](#api)
- [Architecture & Project Structure](#architecture--project-structure)
- [Installation & Usage](#installation--usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)

---

## Business Problem

Banks and FinTech platforms process millions of transactions daily. A small fraction are fraudulent, but manual review of every transaction is impossible at scale. This system automatically scores each transaction, classifies it into a risk band, and generates a human-readable explanation — helping fraud investigation teams prioritize their workload instead of reviewing transactions blindly.

**Important:** This system provides suspicious-transaction detection and alerting support. It does not automatically prevent fraud outright — flagged transactions are recommended for review or block, with a human officer in the loop for medium/high-risk cases.

## System Flow