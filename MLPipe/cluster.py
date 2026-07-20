from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
import pandas as pd 


class cluster: 
    
    def __init__(self,X): 
        self.X = X

    def KMeansPipe(self, optimal_c: int) -> Pipeline: 
        """ 
        KMeans Pipeline is created to perform kmeans clustering analysis.
        Includes minmax-scaling as the main preprocessing method. 
        
        Args: 
            Optimal # of clusters. Retrieve from elbow plot. 
        
        Returns: 
            Pipeline object. Can be used for prediction and analysis. 
        """
        kmeans_pipeline = Pipeline([
        ("MinMax", MinMaxScaler()), 
        ("K-means", KMeans(optimal_c))
        ])

        return kmeans_pipeline

    def elbow_plot(X: pd.DataFrame) -> plt.plot: 
        """ 
        Diagnostic elbow plot used to find the optimal number of clusters.
        Extracts the inertias for kmeans after fitting the model and 
        stores them in a list object. 

        Args:
            X (DataFrame): Input DataFrame to be analysed. 
        
        Returns:
            Elbow Plot for kmeans. 
        
        """
        Minmax = MinMaxScaler() 
        x = Minmax.fit_transform(X)
        inertias = []
        for i in range(1,11): 
            kmeans = KMeans(n_clusters = i)
            kmeans.fit(x)
            inertias.append(kmeans.inertia_)

        plot = plt.plot(range(1,11), inertias, marker='o', linestyle = "--")
        plt.title("Optimal number of Cluster `elbow methdod`")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")

        return plot 
    