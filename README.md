# Single Participant Model

## Overview

The **Single Participant Model** is designed for analyzing and classifying brain data on a per-participant basis. This approach allows for a detailed understanding of individual variations in brain activity and their impact on classification tasks. The model supports two operational modes: **Best Channel** and **Combined Channel**, providing flexibility in feature selection and analysis.

---

## Features

- **Best Channel Mode**: Utilizes the most effective single channel for classification per subject.
- **Combined Channel Mode**: Aggregates data from multiple channels, leveraging a majority-voting approach to enhance performance and robustness.
- **Brain Region Analysis**: Maps effective channels to brain regions and identifies the frequency and consistency of region usage across participants.
- **Customizable Pipelines**: Supports user-defined preprocessing, feature extraction, and classification methods.
- **Visualization Tools**: Includes histograms and plots to analyze the distribution of contributing regions and compare classifier performance.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd single_participant_model
