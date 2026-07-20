from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# fetch dataset 

htru2 = fetch_ucirepo(id=372) 
  
# data (as pandas dataframes) 
X = htru2.data.features 
y = htru2.data.targets 
  
# metadata 
print(htru2.metadata) 
  
# variable information 
print(htru2.variables) 


def pie(X,y) -> plt.plot: 
    df1 = pd.concat([X,y], axis = 1)
    plt.figure(figsize=(10,5))
    # Pie
    plot = plt.pie(y.value_counts(), labels = ["Not Pulsar", "Pulsar"], autopct="%1.0f%%", wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'linestyle': 'solid'})
    plt.title("Proportion of Pulsars")
    return plot

def hist(X,y) -> plt.plot: 
    df1 = pd.concat([X,y], axis = 1)
    plot = plt.hist(data=df1, x = "DM_mean", label = "class")
    plt.title("Distribution of the Average Dispersion Measure")
    plt.xlabel("DM_mean")
    return plot 


def bar(X,y) -> plt.plot: 
    df1 = pd.concat([X,y], axis = 1)
    plot = sns.barplot(df1, x="class", y = "Profile_skewness", palette=["#DC143C", "#0047AB"])
    plt.title("Profile Skewness by Class", fontweight="bold")
    plt.xlabel("Class (1:Pulsar, 0:Non-Pulsar)")
    plt.ylabel("Profile Skewness")
    return plot 

def scatter(X,y) -> plt.plot: 
    df1 = pd.concat([X,y], axis = 1)
    scatter = plt.scatter(x = X["Profile_stdev"], y = X["Profile_mean"], c=y.values, cmap="tab20b", alpha=0.5)
    plt.xlabel("Profile Std-dev")
    plt.ylabel("Profile Mean")

    legend = plt.legend(*scatter.legend_elements(), title="Class")
    plt.title("Profile Mean vs. Std-dev before Balancing", fontweight="bold")
    return scatter 




