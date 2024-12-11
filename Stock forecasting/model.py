import pandas as pd
from datetime import datetime as date, timedelta
import plotly.graph_objs as go
import yfinance as yf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def predict(stock, days_n):
    df = yf.download(stock, period='100d')
    df.reset_index(inplace=True)
    df['Day'] = df.index
    days = [[i] for i in range(len(df.Day))]

    X = days
    Y = df[['Close']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.001, 0.01, 0.1],
            'gamma': [0.001, 0.01, 0.1, 1]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    
    svr_rbf = best_svr
    svr_rbf.fit(x_train, y_train)

    output_days = [[i + x_test[-1][0]] for i in range(1, days_n)]

    future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_n, freq='B')[1:]  # Generate future dates
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=svr_rbf.predict(output_days), mode='lines+markers', name='data'))
    fig.update_layout(title=f"Predicted Close Price of next {days_n - 1} days",
                      xaxis_title="Date", yaxis_title="Closed Price", legend_title="tips")
    return fig
