[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2412.06336)

# A Combined Channel Approach for Decoding Intracranial EEG Signals: Enhancing Accuracy through Spatial Information Integration

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Example](#example)
* [Reading in Data](#reading-in-edf-data)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>
* 
## General Information
............

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2412.06336). Cite this paper using its [DOI](https://doi.org/10.48550/arXiv.2412.06336).

## Getting Started

1. Step 1: Clone the Repository 
`git clone https://github.com/Navid-Ziaei/Combined-Channel-iEEG-Decoder.git`

2. Update Device Path Configuration:
Open config/device_path.yaml and modify the device_path parameter to match your system setup.

3. Set Parameters in setting.yaml:
Configure the necessary parameters in the config/setting.yaml file according to your requirements.

4. Run the Main Script:
Finally, execute the main script to start the program:
`python main.py`

## Repository Structure
This repository is organized as follows:

- `src/main.py`: The main script to run the Single-Patient model.

- `src/data`: Contains scripts for data loading.

- `src/experiments`: Contains scripts for different experiments.

- `src/model`: Contains the Single-Patient model .

- `src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `src/visualization`: Script for result visualization.
<br/>

## Citations
The code contained in this repository for LDGD is companion to the paper:  

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

We encourage you to contribute to LDGD! 

## License

This project is licensed under the terms of the MIT license.
