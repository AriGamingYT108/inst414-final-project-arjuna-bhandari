# inst414-final-project-arjuna-bhandari

Loan Default Prediction Pipeline

--Project Overview--

    This project builds a full end-to-end pipeline to predict loan default risk using historical loan data. We:

    Business problem: Identify borrowers likely to default so lenders can manage risk.

    Data: ~9,500 loan applications with features such as interest rate, FICO score, debt-to-income ratio, loan purpose, and default outcome.

    Techniques: Data extraction (CSV), cleaning and normalization (ETL), feature engineering, Logistic Regression and XGBoost modeling, evaluation (precision, recall, ROC AUC, confusion matrix), and EDA visualizations.

Expected outputs:

    Cleaned dataset (data/extracted/transformed_data.csv)

    Trained model artifacts (.pkl files in data/analysis/models)

    Console-based model evaluation metrics

    Visualization plots (.png files in data/analysis/visualizations)

--Setup Instructions--

Clone the repository:

1. Open the Repo
2. Copy the URL
    Repo: https://github.com/AriGamingYT108/inst414-final-project-arjuna-bhandari.git
3. Clone into VSCode

Create and activate a virtual environment:

1. python -m venv venv
2. venv\Scripts\activate

Install dependencies:

1. pip install -r requirements.txt

-- Run the Project--

1. Execute python main.py
2. OR run each .py seperately in this order:
    1. extract.py
    2. transform_load.py
    3. model.py
    4. evaluate.py
    5. visualize.py

--Code Package Structure--

1. Analysis Folder
    evaluate.py
    model.py
2. Data Folder
    Analysis Folder
        Models Folder
            logistic_regression.pkl
            xgboost.pkl
        Visualizations Folder
            days_wtih_cr_line_distribution.png
            ....
            ....
            ....
            log_annual_inc_distribution.png
    Extracted Folder
        extracted_data.csv
        transformed_data.csv
3. ETL Folder
    extract.py
    transform.py
4. Vis Folder
    visualizations.py
5. Main.py
6. README.MD
7. requirements.txt

