#!/usr/bin/env python3
from etl.extract import extract_data
from etl.transform import transform_load_data
from analysis.model import model_train
from analysis.evaluate import evaluate_models
from vis.visualizations import visualize

def main():
    print("\nStage 1: Extract Data")
    extract_data()
    print("\n Stage 1 Complete")

    print("\nStage 2: Transform & Load")
    transform_load_data()
    print("\n Stage 2 Complete")

    print("\nStage 3: Model Training")
    model_train()
    print("\n Stage 3 Complete")

    print("\nStage 4: Model Evaluation")
    evaluate_models()
    print("\n Stage 4 Complete")

    print("\nStage 5: Visualizations")
    visualize()
    print("\n Stage 5 Complete")

    print("\nAll pipeline stages completed successfully.")

if __name__ == '__main__':
    main()


