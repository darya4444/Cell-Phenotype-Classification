# Cell Phenotype Classification

## Overview

This project is a Streamlit-based web application for automatic classification of cell phenotypes from four-channel immunofluorescence microscopy images using a trained EfficientNet-B0 deep learning model. ML-part in Cell_Phenotype_Classification.ipynb.

The application accepts four fluorescence channels:

- Blue
- Green
- Red
- Yellow

and predicts one of the following cell treatment classes:

- Untreated
- Paclitaxel
- Vorinostat

The interface also displays:

- uploaded fluorescence channels;
- composite fluorescence image;
- predicted class and confidence score;
- class probabilities;
- channel importance estimated using channel occlusion analysis.

---

## Project structure

```
.
├── app.py                 # Streamlit application
├── model.py               # EfficientNet model definition
├── requirements.txt
├── Dockerfile
├── README.md
├── weights/
│   └── best_model2.pth
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your_username>/<repository_name>.git
cd <repository_name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## Running with Docker

Build the Docker image:

```bash
docker build -t cell-classification .
```

Run the container:

```bash
docker run -p 8501:8501 cell-classification
```

Then open:

```
http://localhost:8501
```

---

## Model

- Architecture: EfficientNet-B0
- Input: four-channel fluorescence images (224 × 224)
- Output classes:
  - Untreated
  - Paclitaxel
  - Vorinostat

---

## Technologies

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Docker
