# COMP4030 Traffic Accident Severity Prediction
Coursework for COMP4030 - A data science project focused on predicting traffic accident severity using machine learning techniques.

## Project Overview
This project analyzes traffic accident data to predict accident severity using various machine learning models. The analysis is performed using Python and popular data science libraries.

## Setup Instructions

### Prerequisites
- Python 3.13 or higher
- pip (Python package installer)

### Virtual Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/comp4030-accident-analysis.git
cd comp4030-accident-analysis
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
The project uses the following main packages (as specified in `requirements.txt`):
- numpy (≥1.26.0) - For numerical computations
- pandas (≥2.1.0) - For data manipulation and analysis
- matplotlib (≥3.8.0) - For data visualization
- seaborn (≥0.13.0) - For statistical data visualization
- scikit-learn (≥1.3.0) - For machine learning algorithms
- jupyter (≥1.0.0) - For running Jupyter notebooks
- notebook (≥7.0.0) - For Jupyter notebook interface

## Project Structure
```
comp4030-accident-analysis/
├── COMP4030_Group98_DataScienceProject.ipynb  # Main Jupyter notebook
├── README.md                                  # Project documentation
├── requirements.txt                           # Project dependencies
└── venv/                                     # Virtual environment directory
```

## Usage
1. Ensure your virtual environment is activated:
```bash
source venv/bin/activate  # On macOS/Linux
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `COMP4030_Group98_DataScienceProject.ipynb` in the Jupyter interface

## Deactivating the Environment
When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

