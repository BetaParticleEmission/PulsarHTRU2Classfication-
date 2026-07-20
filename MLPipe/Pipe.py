from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# fetch dataset 

htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
X = htru2.data.features 
y = htru2.data.targets 

cv = KFold(n_splits=5, shuffle=True, random_state=42)

class rfc_Pipeline: 

    def __init___(self, cv): 
        self.cv = cv

    def pipeline_model(self, cv: int) -> Pipeline: 
        smote = SMOTE()

        pipe = Pipeline([
            ("smote", smote),
            ("scaler", StandardScaler()),
            ("RFECV", RFECV(cv=cv, estimator=RandomForestClassifier(n_estimators=200, max_depth=10, max_features="sqrt"))), 
            ("rfc", RandomForestClassifier(n_estimators=200, max_depth = 10, max_features="sqrt"))]
        )
        
        return pipe


if __name__ == "__main__": 
    rfc_Pipeline.pipeline_model() 
    