#!/usr/bin/env python3
import logging
from etl.extract import extract_data
from etl.transform import transform_load_data
from analysis.model import model_train
from analysis.evaluate import evaluate_models
from vis.visualizations import visualize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # log to console
    ]
)

def main():
    logging.info("Stage 1: Extract Data")
    extract_data()
    logging.info("Stage 1 Complete")

    logging.info("Stage 2: Transform & Load")
    transform_load_data()
    logging.info("Stage 2 Complete")

    logging.info("Stage 3: Model Training")
    model_train()
    logging.info("Stage 3 Complete")

    logging.info("Stage 4: Model Evaluation")
    evaluate_models()
    logging.info("Stage 4 Complete")

    logging.info("Stage 5: Visualizations")
    visualize()
    logging.info("Stage 5 Complete")

    logging.info("All pipeline stages completed successfully.")

if __name__ == '__main__':
    main()


