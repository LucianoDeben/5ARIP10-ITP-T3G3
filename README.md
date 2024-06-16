# 5ARIP10 Interdisciplinary Team Project Track 3 Group 3

![Build Status](https://github.com/LucianoDeben/5ARIP10-ITP-T3G3/actions/workflows/ci.yml/badge.svg)[![codecov](https://codecov.io/gh/LucianoDeben/5ARIP10-ITP-T3G3/branch/main/graph/badge.svg)](https://codecov.io/gh/LucianoDeben/5ARIP10-ITP-T3G3)![GitHub](https://img.shields.io/github/license/LucianoDeben/5ARIP10-ITP-T3G3)![black](https://img.shields.io/badge/code%20style-black-000000.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)

## Introduction

This project, conducted in collaboration between Eindhoven University of Technology (TU/e) and Philips, is focused on developing an AI solution for the Image Guided Therapy Challenge on Transarterial Chemoembolization (TACE) procedures. The goal of this project is to enhance the efficiency and accuracy of TACE procedures using advanced AI techniques.

This methodology utilizes supervised learning to train a model. The model maps unenhanced Digital Direct Radiography (DDR) images to vessel-enhanced DRR images. This mapping is achieved by leveraging a 3D vessel network's latent representation. This novel approach offers a way to reduce contrast agent usage while maintaining high visibility of the vessel network even in deforming volumes.

## Getting Started

### Prerequisites

This project is developed using Python and pip for package management. Ensure you have the following installed on your system:

- Python 3.7 or higher

## Project Structure

- `src/`: Contains the source code for the project, including the Streamlit app and AI model training.
- `test/`: Contains unit tests for the project.
- `models/`: Directory for storing AI model files.
- `requirements.txt`: Lists all Python dependencies required for the project.

## Data

This project uses publicly available datasets provided by The Cancer Imaging Archive (TCIA). To access these datasets, please head to the corresponding webpage [TCIA TACE Dataset](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/).

### Installation

To set up a development environment for this project, follow these steps:

1. Clone the repository to your local machine: `git clone https://github.com/LucianoDeben/5ARIP10-ITP-T3G3.git`
2. Navigate to the project directory.
3. Create a virtual environment: `python -m venv env` or `conda create --name env`.
4. Activate the virtual environment: `source env/bin/activate` (Linux/macOS) or `.\env\Scripts\activate` (Windows) or `conda activate myenv` (Conda).
5. Install the required packages: `pip install -r requirements.txt`

### Usage

After setting up the project, you can run the Streamlit demo application in `src`:

1. `cd src`
2. `streamlit run app.py`

### Testing

To run the unit test of the libary use:

1. `cd test`
2. `python -m unittest discover -k 'test.test_*.py'`

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/LucianoDeben/5ARIP10-ITP-T3G3/blob/main/LICENSE) file for details.

## Acknowledgments

We would like to thank the following resources and individuals:

- Our project mentor, Danny Ruijters, for their guidance.
- Philips, for their collaboration and support in this project.
- Gopalakrishnan, Vivek, and Golland, Polina for their work on [fast auto-differentiable digitally reconstructed radiographs](https://link.springer.com/chapter/10.1007/978-3-031-23179-7_1). We utilized their [`DiffDRR`](https://github.com/eigenvivek/DiffDRR?tab=readme-ov-file#user-content-fn-1-aa759ff9097582506ce05933e125ab0a): Auto-differentiable DRR rendering and optimization in PyTorch in our project.

## Authors

This project was developed by:

- Crapels, Dhani - <d.r.m.crapels@student.tue.nl>
- Deben, Luciano - <l.m.deben@student.tue.nl>
- Bierenbroodspot, Sven - <s.a.k.bierenbroodspot@student.tue.nl>
