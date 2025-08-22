# analysis/visualize.py
"""
Generate selected exploratory data analysis (EDA) visualizations and save them to disk.

This script creates the following plots:
  1. Default rate by loan purpose
  2. Distribution of days with a credit line
  3. Distribution of debt-to-income ratio (DTI)
  4. Distribution of FICO credit scores
  5. Distribution of interest rates
  6. Distribution of log-transformed annual income
  7. Distribution of monthly installment amounts

Each plot is saved under 'data/analysis/visualizations'.

Usage:
    python analysis/visualize.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def visualize():
    """
    Loads the cleaned dataset, generates specified histograms and count plots,
    and writes them as PNG files for review.
    """
    # Load the transformed data
    df = pd.read_csv('data/extracted/lending_cleaned.csv')
    # Normalize header names to snake_case without suffixes
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    # Prepare output directory for visualizations
    vis_dir = 'data/analysis/visualizations'
    if os.path.exists(vis_dir):
        # Clear existing images
        for file in os.listdir(vis_dir):
            os.remove(os.path.join(vis_dir, file))
    else:
        os.makedirs(vis_dir, exist_ok=True)

    # 1. Default by Loan Purpose
    # Shows the count of loans by purpose, split by default status (1=default, 0=fully paid)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='purpose', hue='not_fully_paid', data=df)
    plt.title('Default by Loan Purpose')
    plt.xlabel('Loan Purpose')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'default_by_purpose.png'))
    plt.close()

    # 2. Distribution of Days with Credit Line
    # Histogram of how many days each borrower has had an open credit line
    plt.figure(figsize=(8, 5))
    sns.histplot(df['days_with_cr_line'], kde=True)
    plt.title('Distribution of Days with Credit Line')
    plt.xlabel('Days with Credit Line')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'days_with_cr_line_distribution.png'))
    plt.close()

    # 3. Distribution of Debt-to-Income Ratio (DTI)
    # Shows how borrowers' debt compares to their income
    plt.figure(figsize=(8, 5))
    sns.histplot(df['dti'], kde=True)
    plt.title('Distribution of Debt-to-Income Ratio')
    plt.xlabel('DTI')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'dti_distribution.png'))
    plt.close()

    # 4. Distribution of FICO Scores
    # Illustrates the spread of borrowers' credit scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df['fico'], kde=True)
    plt.title('Distribution of FICO Scores')
    plt.xlabel('FICO Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fico_distribution.png'))
    plt.close()

    # 5. Distribution of Interest Rate
    # Histogram of the interest rates applied to loans
    plt.figure(figsize=(8, 5))
    sns.histplot(df['int_rate'], kde=True)
    plt.title('Distribution of Interest Rate')
    plt.xlabel('Interest Rate')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'int_rate_distribution.png'))
    plt.close()

    # 6. Distribution of Log Annual Income
    # Visualizes borrowers' log-transformed annual incomes
    plt.figure(figsize=(8, 5))
    sns.histplot(df['log_annual_inc'], kde=True)
    plt.title('Distribution of Log Annual Income')
    plt.xlabel('Log Annual Income')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'log_annual_inc_distribution.png'))
    plt.close()

    # 7. Distribution of Installment Amounts
    # Histogram of monthly payment amounts for each loan
    plt.figure(figsize=(8, 5))
    sns.histplot(df['installment'], kde=True)
    plt.title('Distribution of Installment Amounts')
    plt.xlabel('Installment Amount')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'installment_distribution.png'))
    plt.close()

    logging.info("All visualizations saved to %s", vis_dir)



if __name__ == '__main__':
    visualize()
