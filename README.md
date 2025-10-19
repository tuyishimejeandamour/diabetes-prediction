# Diabetes Risk Prediction Using Machine Learning

A comprehensive machine learning project that predicts diabetes risk using the CDC's BRFSS dataset. This project compares traditional ML algorithms with deep learning approaches to identify individuals at risk of diabetes.

## Dataset

The project uses the BRFSS (Behavioral Risk Factor Surveillance System) dataset containing health information from 253,680 individuals with 21 features including BMI, age, general health status, and lifestyle factors.

## Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

## Project Structure

```
diabetes/
├── datasets/
│   └── diabetes.csv          # CDC BRFSS dataset
├── diabetes.ipynb            # Main Jupyter notebook
└── README.md                 # This file
```

## How to Run

### Option 1: Using Jupyter Notebook (Recommended)

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /home/damour/Documents/alu/diabetes
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**:
   - In your browser, click on `diabetes.ipynb`
   - Run cells sequentially from top to bottom using `Shift + Enter`
   - Or run all cells at once: `Cell` → `Run All`

### Option 2: Using VS Code

1. **Open the project folder** in VS Code
2. **Open** `diabetes.ipynb`
3. **Select a Python kernel** (click on the kernel selector in the top right)
4. **Run cells** by clicking the play button next to each cell or use `Shift + Enter`

### Option 3: Using JupyterLab

1. **Install JupyterLab**:
   ```bash
   pip install jupyterlab
   ```

2. **Launch JupyterLab**:
   ```bash
   jupyter lab
   ```

3. **Open and run** `diabetes.ipynb`

## What the Notebook Does

The notebook performs the following analysis:

1. **Data Exploration**: Load and analyze the diabetes dataset
2. **Data Preprocessing**: Handle class imbalance, feature scaling, and train-test split
3. **Traditional ML Models**: Test 5 different algorithms:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors
   - Decision Tree

4. **Deep Learning Experiments**: 7 different neural network architectures:
   - Baseline Neural Network
   - Deeper Network with Dropout
   - Batch Normalization
   - Functional API with Skip Connections
   - SGD with Momentum
   - Wide Network Architecture
   - L2 Regularization with RMSprop

5. **Model Evaluation**: Compare models using:
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices
   - Feature importance analysis

## Expected Runtime

- **Complete notebook**: ~15-30 minutes (depending on your hardware)
- **Traditional ML section**: ~2-5 minutes
- **Deep Learning section**: ~10-20 minutes

## Key Findings

- Best performing models achieve ROC-AUC scores of 0.80-0.85
- Most important predictive features: BMI, Age, General Health Status
- Both traditional ML and deep learning approaches are effective
- Regularization techniques (dropout, batch normalization) improve performance

## Troubleshooting

### TensorFlow Issues
If you encounter TensorFlow errors:
```bash
pip install --upgrade tensorflow
```

### Memory Issues
If running out of memory, reduce batch sizes or use fewer experiments.

### Missing Dataset
Ensure `datasets/diabetes.csv` is in the correct location.

## Author

This project demonstrates healthcare analytics using machine learning to predict diabetes risk and support early intervention strategies.

## License

Educational project for learning purposes.
