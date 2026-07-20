import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from imblearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo 
import plotly.express as px
from imblearn.over_sampling import SMOTE


htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
X = htru2.data.features 
y = htru2.data.targets 

sm = SMOTE(random_state=42) 
X_resampled, y_resampled = sm.fit_resample(X,y)

df = pd.concat([X_resampled, y_resampled], axis=1)

class Stats: 

    def __init__(self, df, X, y): 
        self.df = df 
        self.X = X 
        self.y = y 

    def corr_coef(self): 

        cor_matrix = self.df.corr()

        targ_cor = cor_matrix[["class"]].sort_values(by="class", ascending=False)
        cor_plot = sns.heatmap(targ_cor, cmap = "Blues", annot=True, linecolor="white", linewidths= 1)
        plt.title("Correlations")
        return cor_plot 
    
    def multicollinearity(self, X = X):
        """ 
        Computes the multicollinearity using the statsmodels package. 

        Args: 
            X predictor dataframe.  
        
        Returns: 
            VIF values for each of the predictor variables. 
        
        """

        vif_data = pd.DataFrame({}) 
        vif_data["Feature"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, idx) for idx in range(self.X.shape[1])]
        return vif_data
    
    def PCA_test(self): 
        """ 
        Perform PCA and plot the explained variance ratio
        using the cumulative sum and training-testing split. 

        Args:
            None. 

        Returns: 
            Matplotlib plot containing the explained variance ratio across 'n' components. 
        
        
        """


        scaler = StandardScaler()

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        x_train_scaled = scaler.fit_transform(x_train)
        pca = PCA().fit(x_train_scaled, y_train)
        plot = plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("Explained Variance from Components")
        plt.xlabel("Number of components")
        plt.ylabel("Cumsum of Explained Variance ")
        
        return plot 
