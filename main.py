#!/usr/bin/env python3
from etl.extract import extract_data
from etl.transform import transform_load_data
from analysis.model import model_train
from analysis.evaluate import evaluate_models
from vis.visualizations import visualize

def main():
    print("\n=== Stage 1: Extract Data ===")
    extract_data()

    print("\n=== Stage 2: Transform & Load ===")
    transform_load_data()

    print("\n=== Stage 3: Model Training ===")
    model_train()

    print("\n=== Stage 4: Model Evaluation ===")
    evaluate_models()

    print("\n=== Stage 5: Visualizations ===")
    visualize()

    print("\nAll pipeline stages completed successfully.")

if __name__ == '__main__':
    main()
