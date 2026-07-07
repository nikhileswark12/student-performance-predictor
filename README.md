# Student Performance Predictor

This project is a beginner-level machine learning notebook designed to predict student test scores across math, reading, and writing based on demographic and background factors. The primary goal is to walk through a complete, simple, and readable end-to-end data science workflow without relying on complex optimizations, advanced pandas chaining, or complicated modeling frameworks. It serves as an accessible introduction to data processing and regression analysis.

## Vision

To provide a clear, step-by-step example of how basic data cleaning, exploratory data analysis, and predictive modeling can be applied to real-world educational datasets using standard Python libraries, helping beginners grasp foundational ML concepts.

## Problem Statement

Educational outcomes are influenced by a variety of background factors, including parental education, lunch plans, and test preparation courses. Identifying which factors most strongly correlate with student success and building simple predictive models can help educators understand potential disparities and target interventions. This project models student performance across multiple subjects to see how well these outcomes can be predicted using standard regression techniques.

## Key Features

- **Exploratory Data Analysis**: Visualizes correlations, distributions, and the impact of test preparation on performance using clean, interpretable plots.
- **Data Preprocessing**: Handles categorical encoding straightforwardly using standard pandas functions.
- **Multi-Target Modeling**: Trains and evaluates models separately for math, reading, writing, and an aggregated average score.
- **Model Comparison**: Benchmarks Linear Regression against Decision Trees and Random Forests using 5-fold cross-validation mean R² and standard deviation.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Primary programming language |
| **Jupyter** | Interactive notebook environment |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical computations |
| **Matplotlib** | Foundational data visualization |
| **Seaborn** | Statistical data visualization |
| **scikit-learn** | Machine learning model training and evaluation |

## System Architecture

The system consists of a single interactive Jupyter Notebook that loads a local CSV file, processes the data in memory, trains three different regression models for four distinct targets, and outputs performance metrics and visualizations inline. The best-performing model for each target is then saved to disk for potential reuse.

## ML Pipeline

1. **Import Libraries**: Load standard data science and machine learning packages.
2. **Load Data**: Read the dataset into a pandas DataFrame and examine its structure.
3. **Exploratory Data Analysis**: Visualize key relationships and score distributions.
4. **Data Cleaning**: Verify the absence of missing values and duplicate records.
5. **Feature Engineering**: One-hot encode categorical variables and calculate an average score.
6. **Train-Test Split**: Partition the dataset uniformly into 80% training and 20% testing sets.
7. **Train Models**: Fit Linear Regression, Decision Tree, and Random Forest models with default parameters.
8. **Evaluate and Compare**: Evaluate models using 5-fold cross-validation and report mean R² and standard deviation.
9. **Feature Importance**: Extract and plot the most influential features from the Random Forest model.
10. **Best Model Selection**: Identify the highest-performing model based on R² for each target variable.
11. **Save Best Models**: Save the best-performing model for each target to disk as a `.joblib` file for potential reuse.

## Model Performance

*Note: Linear Regression consistently outperforms the tree-based models across all targets, which suggests that the underlying relationships in this dataset are close to linear. To prevent overfitting on a dataset of this size, the Decision Tree and Random Forest models were heavily regularized by capping their depth (`max_depth=5`).*

### Math Score
| Model | Cross-Validated R² (Mean ± Std) |
|-------|---------------------------------|
| Linear Regression | 0.2279 ± 0.0424 |
| Decision Tree | 0.1090 ± 0.0540 |
| Random Forest | 0.1666 ± 0.0414 |

### Reading Score
| Model | Cross-Validated R² (Mean ± Std) |
|-------|---------------------------------|
| Linear Regression | 0.1937 ± 0.0613 |
| Decision Tree | 0.0688 ± 0.1063 |
| Random Forest | 0.1312 ± 0.0779 |

### Writing Score
| Model | Cross-Validated R² (Mean ± Std) |
|-------|---------------------------------|
| Linear Regression | 0.3093 ± 0.0488 |
| Decision Tree | 0.1937 ± 0.0830 |
| Random Forest | 0.2485 ± 0.0706 |

### Average Score
| Model | Cross-Validated R² (Mean ± Std) |
|-------|---------------------------------|
| Linear Regression | 0.2123 ± 0.0465 |
| Decision Tree | 0.0792 ± 0.0901 |
| Random Forest | 0.1430 ± 0.0640 |

## Notebook Sections

1. Import Libraries
2. Load Data
3. Exploratory Data Analysis
4. Data Cleaning
5. Feature Engineering
6. Train-Test Split
7. Train Models
8. Evaluate and Compare
9. Feature Importance
10. Best Model Selection

## Dataset

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Size**: 1,000 records
- **Target Variables**: Math Score, Reading Score, Writing Score, Average Score

## Design Principles

- **Simplicity First**: Code is written linearly with clear variable names and minimal pandas chaining.
- **No Complex Encoders**: Standard `pd.get_dummies` is used for all categorical features.
- **Readable Structure**: Strict adherence to one markdown title and one code block per section.
- **Self-Contained Execution**: Designed to be run top-to-bottom sequentially without external dependencies or persisting state.

## Project Structure

```text
student-performance-predictor/
├── models/
│   ├── average_score_model.joblib
│   ├── math_score_model.joblib
│   ├── reading_score_model.joblib
│   └── writing_score_model.joblib
├── README.md
├── requirements.txt
├── StudentPerformance.ipynb
└── StudentsPerformance.csv
```

## Setup Instructions

1. Ensure Python 3 is installed.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Launch Jupyter Notebook from your activated virtual environment:
```bash
jupyter notebook
```
Open `StudentPerformance.ipynb` and run all cells from top to bottom.

## Current Status

- **Completed**: End-to-end data processing, EDA, modeling, 5-fold cross-validation evaluation, tree depth-capping, and model persistence.
- **Planned Enhancements**:
  - Hyperparameter Tuning