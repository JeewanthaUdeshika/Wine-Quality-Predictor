'''
CO544 - Machine Learning and Data Mining
Lab 03

Ariyawansha P.H.J.U.
E/18/028
'''

# For Data Analysis
import pandas as pd

# For model creation and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

# For visualizations and interactive dashboard creation
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash_bootstrap_components as dbc

# Load dataset
data = pd.read_csv('data/winequality-red.csv')
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first')
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
    # Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']


# Split the dat a into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create an instance of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server
# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.Div(
            style={'background-color': '#e6bcad'},
            children=[
                html.Div(
                    style={'padding': '10px', 'text-align': 'center'},
                    children=[
                        html.H3('CO544-2023 Lab 3'),
                        html.H1('Wine Quality Prediction', style={'font-family': 'Arial'}),
                    ]
                ),
                html.Div(
                    style={'display': 'flex', 'justify-content': 'center', 'background-color': 'white'},
                    children=[
                        html.H4('© Jeewantha Ariyawansha')
                    ]
                )
            ]
        ),
        
        html.H2('Exploratory Data Analysis', style={'justify-content': 'center', 'font-family': 'Arial'}),
        html.Div([
            # Layout for exploratory data analysis: correlation between two selected features
            html.Div([
                html.Label('Feature 1 (X-axis)'),
                    dcc.Dropdown(
                    id='x_feature',
                    options=[{'label': col, 'value': col} for col in data.columns],
                    value=data.columns[0]
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Feature 2 (Y-axis)'),
                dcc.Dropdown(
                    id='y_feature',
                    options=[{'label': col, 'value': col} for col in data.columns],
                    value=data.columns[1]
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            dcc.Graph(id='correlation_plot'),
        ], style={'background-color': '#ade6d8', 'margin':'0 0 30px 0'}),

        html.H2('Wine Quality Prediction', style={'justify-content': 'center', 'font-family': 'Arial'}),
        html.Div([
            # Layout for wine quality prediction based on input feature values

            html.Div([
                html.Label("Fixed Acidity"),
                dcc.Input(id='fixed_acidity', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),

                html.Label("Volatile Acidity"),
                dcc.Input(id='volatile_acidity', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Citric Acid"),
                dcc.Input(id='citric_acid', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Residual Sugar"),
                dcc.Input(id='residual_sugar', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Chlorides"),
                dcc.Input(id='chlorides', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Free Sulfur Dioxide"),
                dcc.Input(id='free_sulfur_dioxide', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Total Sulfur Dioxide"),
                dcc.Input(id='total_sulfur_dioxide', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Density"),
                dcc.Input(id='density', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("pH"),
                dcc.Input(id='ph', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Sulphates"),
                dcc.Input(id='sulphates', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
                
                html.Label("Alcohol"),
                dcc.Input(id='alcohol', type='number', required=True, style={'margin':'10px 10px 10px 10px'}),
                html.Br(),
            ]),
            html.Div([
                html.Button('Predict', id='predict-button', n_clicks=0),
        ]),
            html.Div([
                html.H4("Predicted Quality"),
                html.Div(id='prediction-output')
            ])   
                ], style={'background-color': '#ade6d8', 'margin':'0 0 30px 0'}),

])

# Adding Interactivity
# Define the callback to update the correlation plot
import numpy as np
import logging

# Configure the logging
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.callback(
    dash.dependencies.Output('correlation_plot', 'figure'),
    [dash.dependencies.Input('x_feature', 'value'),
     dash.dependencies.Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x = x_feature, y=y_feature, color='quality')
    fig.update_layout(title=f'Correlation between {x_feature} and {y_feature}')
    return fig

# Define the callback function to predict wine quality
@app.callback(
    Output(component_id = 'prediction-output', component_property ='children'),
    [dash.dependencies.Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('fixed_acidity', 'value'),
     dash.dependencies.State('volatile_acidity', 'value'),
     dash.dependencies.State('citric_acid', 'value'),
     dash.dependencies.State('residual_sugar', 'value'),
     dash.dependencies.State('chlorides', 'value'),
     dash.dependencies.State('free_sulfur_dioxide', 'value'),
     dash.dependencies.State('total_sulfur_dioxide', 'value'),
     dash.dependencies.State('density', 'value'),
     dash.dependencies.State('ph', 'value'),
     dash.dependencies.State('sulphates', 'value'),
     dash.dependencies.State('alcohol', 'value')]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    # Create input features array for prediction
    input_features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]).reshape(1,-1)

    logging.info(input_features)

    

    # if (n_clicks>0):
    # Predict the wine quality
    prediction = logreg_model.predict(input_features)[0]

        # Log the inputs and outputs
    logging.info("Input 1: {}".format(fixed_acidity))
    logging.info("Input 2: {}".format(volatile_acidity))
    logging.info("Output: {}".format(prediction))

    # Return the prediction
    if prediction == 1:
        return 'This wine is predicted to be good quality'
    else:
        return 'This wine is predicted to be bad quality'
    

# Running the Dashboard
if __name__ == '__main__':
    app.run_server(debug=False)