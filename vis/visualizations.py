import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize():
    """
    Generate selected EDA plots and save to data/analysis/visualizations:
      - default_by_purpose.png
      - days_with_cr_line_distribution.png
      - dti_distribution.png
      - fico_distribution.png
      - int_rate_distribution.png
      - log_annual_inc_distribution.png
      - installment_distribution.png
    """
    df = pd.read_csv('data/extracted/transformed_data.csv')
    df.columns = [c.split()[0].strip().lower().replace('.', '_') for c in df.columns]

    vis_dir = 'data/analysis/visualizations'
    if os.path.exists(vis_dir):
        for f in os.listdir(vis_dir):
            os.remove(os.path.join(vis_dir, f))
    else:
        os.makedirs(vis_dir, exist_ok=True)

    # 1. Default by Purpose
    plt.figure(figsize=(8,5))
    sns.countplot(x='purpose', hue='not_fully_paid', data=df)
    plt.title('Default by Loan Purpose')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/default_by_purpose.png")
    plt.close()

    # 2. Days with Credit Line
    plt.figure(figsize=(8,5))
    sns.histplot(df['days_with_cr_line'], kde=True)
    plt.title('Distribution of Days with Credit Line')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/days_with_cr_line_distribution.png")
    plt.close()

    # 3. DTI
    plt.figure(figsize=(8,5))
    sns.histplot(df['dti'], kde=True)
    plt.title('Distribution of Debt-to-Income Ratio')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/dti_distribution.png")
    plt.close()

    # 4. FICO
    plt.figure(figsize=(8,5))
    sns.histplot(df['fico'], kde=True)
    plt.title('Distribution of FICO Scores')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/fico_distribution.png")
    plt.close()

    # 5. Interest Rate
    plt.figure(figsize=(8,5))
    sns.histplot(df['int_rate'], kde=True)
    plt.title('Distribution of Interest Rate')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/int_rate_distribution.png")
    plt.close()

    # 6. Log Annual Income
    plt.figure(figsize=(8,5))
    sns.histplot(df['log_annual_inc'], kde=True)
    plt.title('Distribution of Log Annual Income')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/log_annual_inc_distribution.png")
    plt.close()

    # 7. Installment
    plt.figure(figsize=(8,5))
    sns.histplot(df['installment'], kde=True)
    plt.title('Distribution of Installment Amounts')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/installment_distribution.png")
    plt.close()

    print(f"Selected visualizations saved to {vis_dir}")

if __name__ == '__main__':
    visualize()
