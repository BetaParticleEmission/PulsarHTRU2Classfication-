from dash import html, Dash, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd 

# Import Data 
from ucimlrepo import fetch_ucirepo 

htru2 = fetch_ucirepo(id=372) 
   
X = htru2.data.features 
y = htru2.data.targets 
df = pd.concat([X,y], axis=1)
# Initialize App


app = Dash() 

app.layout = [
    html.H1(children="Pulsar Predictior", style={'textAlign':'center'}),
    dcc.Dropdown(["Profile_mean", "DM_mean", "class"], 'Select', id="dropdown-selection"),
    dcc.Graph(id="graph-content")
    
]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value): 
    col = str(value)
    cols = list(df.columns)
    if col in cols[0]: 
        scatter = px.scatter(df, x = "Profile_stdev", y = "Profile_mean", color='class', labels = {0: "Not Pulsar", 1:"Pulsar", "class":"Star Type"}, title = "Profile Mean vs. Std-dev",)
        return (scatter)
    elif col in cols[4]: 
        hist = px.histogram(df, x = "DM_mean")
        return (hist)
    elif col in cols[-1]: 
        pie = px.pie(df, names="class",title="Proportion of Star Type")    
        pie = pie.update_traces(marker_line_color="black", marker_line_width = 2)
        return(pie)

if __name__ == '__main__':
    app.run(debug=True)