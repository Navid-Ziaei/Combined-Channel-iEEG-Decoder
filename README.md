[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2412.06336)

# A Combined Channel Approach for Decoding Intracranial EEG Signals: Enhancing Accuracy through Spatial Information Integration

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>

## General Information
The Single Participant Model is designed to analyze and classify brain data on a per-participant basis. This approach enables a detailed understanding of individual variations in brain activity and their impact on classification tasks. The model supports two operational modes: Best Channel and Combined Channel, providing flexibility in feature selection and analysis.

### Features
- **Best Channel Mode**: Utilizes the most effective single channel for classification per subject.
- **Combined Channel Mode**: Aggregates data from multiple channels, leveraging a majority-voting approach to enhance performance and robustness.
- **Brain Region Analysis**: Maps effective channels to brain regions and identifies the frequency and consistency of region usage across participants.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2412.06336). Cite this paper using its [DOI](https://doi.org/10.48550/arXiv.2412.06336).

## Getting Started

1. Clone the Repository 
`git clone https://github.com/Navid-Ziaei/Combined-Channel-iEEG-Decoder.git`

2. Update Device Path Configuration:
Open `config/device_path.yaml` and modify the device_path parameter to match your system setup.

3. Set Parameters in setting.yaml:
Configure the necessary parameters in the `config/setting.yaml` file according to your requirements.

4. Run the Main Script:
Finally, execute the main script to start the program:
`python main.py`

## Repository Structure
This repository is organized as follows:

- `src/main.py`: The main script to run the Single-Patient model.

- `src/data`: Contains scripts for data loading.

- `src/experiments`: Contains scripts for different experiments.

- `src/feature_extraction`: Contains scripts for feature extraction methods.

- `src/model`: Contains the Single-Patient model .

- `src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `src/visualization`: Script for result visualization.
<br/>

## Citations
The code contained in this repository for Single-Patient model is companion to the paper:  

```
@article{memar2024combined,
  title={A Combined Channel Approach for Decoding Intracranial EEG Signals: Enhancing Accuracy through Spatial Information Integration},
  author={Memar, Maryam Ostadsharif and Ziaei, Navid and Nazari, Behzad},
  journal={arXiv preprint arXiv:2412.06336},
  year={2024}
}
```
which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to Single-Patient model! 

## License

This project is licensed under the terms of the MIT license.
