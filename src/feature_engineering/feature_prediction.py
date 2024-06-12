import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Sklearn Imports
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score

# Monolithic Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Custom Imports
from utility.config import logger, OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME
import utility.utils as utils

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
extract_dir = os.path.join(current_dir)
sys.path.append(extract_dir)

def discretize_target(y, n_bins=3):
    """Convert the target variable into a discrete variable, transforming it into a classification problem."""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    y_binned = est.fit_transform(y.reshape(-1, 1)).astype(int).ravel()
    return y_binned

def transformed_predictive_features(X):
    """Apply various transformations to the features."""
    logger.info("Applying transformations to the features")
    transformations = {}

    # Standardization
    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X)
    transformations["Standardization"] = X_standardized

    # Normalization
    scaler_minmax = MinMaxScaler()
    X_normalized = scaler_minmax.fit_transform(X)
    transformations["Normalization"] = X_normalized

    # Dimensionality reduction
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_standardized)
    transformations["Dimensionality reduction"] = X_pca

    # Polynomial features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    transformations["Polynomial features"] = X_poly

    # Binning or discretization
    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X_binned = est.fit_transform(X)
    transformations["Binning or discretization"] = X_binned

    transformations["Original"] = X
    return transformations

def get_classifiers():
    """Return a list of classifiers."""
    classifiers = [
        ("DecisionTree", DecisionTreeClassifier(criterion='gini', min_samples_split=10)),
        ("KNN", KNeighborsClassifier(n_neighbors=10, p=1, algorithm='ball_tree', weights='distance')),
        ("NaiveBayes", GaussianNB(var_smoothing=1e-6)),
        ("SVM", SVC(probability=True)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(8), activation='identity', max_iter=100, solver='adam', tol=1e-3, early_stopping=True, validation_fraction=0.3))
    ]
    return classifiers

def perform_iteration_over_transformations(transformations, clf, y_discretized, folds=5):
    """Perform iteration over transformations and return the results."""
    clf_results = {}

    for name, X_transformed in transformations.items():
        result = model_selection.cross_val_score(clf, X_transformed, y_discretized, cv=folds)
        Z = model_selection.cross_val_predict(clf, X_transformed, y_discretized, cv=folds)
        f1Score = f1_score(y_discretized, Z, pos_label=0, average='weighted')
        cm = confusion_matrix(y_discretized, Z)

        clf_results[name] = {
            "accuracy": result.mean(),
            "std": result.std(),
            "f1_score": f1Score,
            "confusion_matrix": cm
        }

        logger.info(f"Results based on Cross Validation with {name}:")
        logger.info(f"Accuracy: {result.mean():.5f}")
        logger.info(f"Standard deviation: {result.std():.5f}")
        logger.info(f"F1_score: {f1Score:.5f}")
        logger.debug(f"Confusion Matrix:")
        logger.debug(f"\n{cm}")
        logger.info("="*70)

    return clf_results

def get_best_classifier(results, metric):
    """Return the classifier with the highest accuracy or F1-score."""
    if not results:
        logger.error("No results to evaluate.")
        return None
    return max(results, key=lambda x: results[x][max(results[x], key=lambda y: results[x][y][metric])][metric])

def show_best_results(results, metric):
    """Show the best results."""
    if not results:
        logger.error("No results to evaluate.")
        return None
    max_clf = get_best_classifier(results, metric)
    if max_clf:
        logger.info(f"Classifier with highest {metric}: {max_clf}")
        logger.info(f"{metric}: {results[max_clf][max(results[max_clf], key=lambda y: results[max_clf][y][metric])][metric]:.5f}")
    return max_clf

def apply_feature_prediction_classification():
    logger.info("Applying classification")
    
    # Read the filtered merged file
    df = utils.read_csv_file_as_dataframe(utils.join_file_path(OUTPUT_MERGED_PATH, OUTPUT_FILTERED_MERGED_FILENAME))
    
    # Assuming the last column is the target variable (val_fatorcapacidade)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Discretize the target variable
    y_discretized = discretize_target(y)

    # Get transformed features
    transformations = transformed_predictive_features(X)

    folds = 5
    results = {}

    # Iteration over classifiers
    for clf_name, clf in get_classifiers():
        logger.info("="*70)
        logger.info(f"Classifier: {clf_name}")
        clf_results = perform_iteration_over_transformations(transformations, clf, y_discretized, folds)
        results[clf_name] = clf_results

    # Show the results
    show_best_results(results, "accuracy")
    show_best_results(results, "f1_score")
    
    logger.info("="*70)
