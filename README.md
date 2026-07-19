# Student Performance Predictor

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20GridSearchCV-orange)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-red)

This project is a beginner-friendly yet highly professional machine learning pipeline designed to predict student test scores (specifically, the overall average score) based on demographic and background factors. 

The primary goal is to walk through a complete, robust data science workflow—from exploratory data analysis to a fully leakage-free preprocessing pipeline and hyperparameter tuning using GridSearchCV.

## 📌 Problem Statement

Educational outcomes are shaped by a complex interplay of various factors, such as parental education, socioeconomic status (approximated by lunch programs), and access to test preparation. By building a machine learning model, we aim to uncover which of these factors most strongly correlate with student success and how accurately we can predict these educational outcomes. Understanding these patterns could help educators tailor interventions to support students more effectively.

## 🚀 Key Features

- **Explicit Model Training**: Instead of hiding behind complex pipelines or grid search loops, **Linear Regression**, **Decision Tree**, and **Random Forest** models are explicitly instantiated and manually trained from `scikit-learn`. This clear, unrolled structure is perfect for beginners to see exactly what `.fit()` and `.predict()` are doing.

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Primary programming language |
| **Jupyter** | Interactive notebook environment |
| **Pandas & NumPy** | Data manipulation and numerical computations |
| **Matplotlib & Seaborn**| Statistical data visualization |
| **scikit-learn** | Preprocessing, Modeling, and Evaluation |

## 🏗️ ML Workflow

The system is contained within a single interactive Jupyter Notebook that executes the following workflow:

1. **Load Data**: Read the dataset into a pandas DataFrame.
2. **EDA & Data Cleaning**: Visualize key relationships and verify the absence of missing values and duplicate records.
3. **Feature Engineering**: Create an aggregated average score target. *Categorical features are left as raw strings to be handled by the pipeline.*
4. **Train-Test Split**: Partition the dataset (80% training, 20% testing).
5. **Preprocessing & Training Pipeline**: Construct a `ColumnTransformer` to encode categorical variables, bundled with regression models into a `Pipeline`.
6. **Hyperparameter Tuning**: Optimize model parameters using `GridSearchCV` with 5-fold cross-validation.
7. **Evaluate**: Assess models on the unseen test set using MAE, RMSE, and R² metrics.
8. **Feature Importance**: Extract and plot the most influential features.
9. **Save Models**: Persist the best end-to-end `Pipeline`.

## 📊 Model Performance

After hyperparameter tuning, the models were evaluated on the unseen test set. Linear Regression performed exceptionally well, indicating that the underlying relationships in the dataset are largely linear. 

| Target | Best Model | Test R² | Test RMSE |
|--------|------------|---------|-----------|
| **Average** | Linear Regression | 0.1622 | 13.40 |

*Note: The Tree-based models (Decision Tree, Random Forest) were heavily tuned using Grid Search to prevent overfitting on this 1,000-record dataset.*

## 📁 Dataset

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams)
- **Size**: 1,000 records, 8 features
- **Target Variable**: Average Score

## 📂 Project Structure

```text
student-performance-predictor/
├── models/                         # Saved Pipeline artifacts (.joblib)
│   └── best_average_score_model.joblib
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── StudentPerformance.ipynb        # Main workflow notebook
└── StudentsPerformance.csv         # Dataset
```

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nikhileswark12/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `StudentPerformance.ipynb` and run all cells sequentially.

## 🔮 Future Improvements

- Evaluate advanced algorithms like Gradient Boosting (XGBoost, LightGBM).
- Build a lightweight web application (e.g., using Streamlit or Flask) to serve the `.joblib` pipelines for real-time predictions.
- Collect more expansive educational datasets to improve model generalization.