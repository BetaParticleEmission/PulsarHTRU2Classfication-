# Data Import
from ucimlrepo import fetch_ucirepo

# Data Manipulation and plots
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

# Numeric and Statistical 
import statsmodels.api as sm
import numpy as np 

# ML Packages 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
  
# Custom 
from MLPipe.Pipe import rfc_Pipeline
from Analysis.plots import pie,hist,bar,scatter


def main(): 

    # Predictors with target
    htru2 = fetch_ucirepo(id=372) 
    X = htru2.data.features 
    y = htru2.data.targets 

    # Data Visualization 

    _, ax = plt.subplots(4)
    pie(X,y)
    bar(X,y)
    hist(X,y)
    scatter(X,y)


    # ML Pipeline

    with mlflow.start_run(): 

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        rfc = rfc_Pipeline() 
        pipe = rfc.pipeline_model(cv = cv) 
        pipe.fit(x_train, y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel())

        # Log Model Hyperparameters 
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("max_features", "sqrt")

        pred = pipe.predict(x_test)
        rfc = pipe.named_steps["rfc"]

        # Metrics 

        final_feature_names = pipe[:-1].get_feature_names_out()

        rfc_feature_importances = pd.DataFrame({"Importance":rfc.feature_importances_, "Labels":final_feature_names})

        
        print(rfc_feature_importances)
        print(f"Accuracy Score:{accuracy_score(y_test, pred)}") 


        mlflow.log_metric("Accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("F1-Score", f1_score(y_test, pred))


        cm = confusion_matrix(y_test, pred, labels = pipe.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=pipe.classes_) 
        disp.plot()
        plt.show() 

        # Log Model 

        mlflow.sklearn.log_model(rfc, "Random Forest Classifier")




if __name__ == "__main__": 
        main()  