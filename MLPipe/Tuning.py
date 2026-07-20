from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 

htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
X = htru2.data.features 
y = htru2.data.targets 
  

def HyperTune(X,y):    
    """ 
    Selects the best hyperparameters for the classification model.

    Args: 
        X and Y (DataFrame). 
    
    Returns: 
        Best hyperparamters for the model. 

    
    """
    scaler = StandardScaler() 
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42) 
    x_train_scaled = scaler.fit_transform(x_train)

    rfc = RandomForestClassifier()
    param_set = {
        'n_estimators': [100,200],
        "max_features": ["sqrt",None],
        "max_depth": [None,10,20],
    }
    gs = GridSearchCV(estimator=rfc, param_grid=param_set,cv=5,n_jobs=-1)
    gs.fit(x_train_scaled, y_train)

    return print(f"Best Model Hyperparamters: {gs.best_params_}")