import sys
import os

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

from utility.config import logger

def visualize_pca_components():
    logger.info("Visualizing PCA components")

def apply_pca_analysis():
    logger.info("Applying PCA Analysis")
    visualize_pca_components()
