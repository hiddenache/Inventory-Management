import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from django.shortcuts import render

def prediction_view(request):
    vanzari = pd.read_csv("vanzari.csv")
    vanzari['Month'] = pd.to_datetime(vanzari['Month'])
    vanzari['Month'] = vanzari['Month'].dt.to_period('M')

    # grupam vanzarile dupa luna
    vanzari_lunare = vanzari.groupby('Month')['Orders'].sum().reset_index()

    # diferentele dintre vanzarile lunare
    vanzari_lunare['Month'] = vanzari_lunare['Month'].dt.to_timestamp()
    vanzari_lunare['dif_comenzi'] = vanzari_lunare['Orders'].diff()
    vanzari_lunare = vanzari_lunare.dropna()

    # nu avem nevoie de vanzari si data vanzarilor

    diferenta_vanzari = vanzari_lunare.drop(['Month', 'Orders'], axis=1)

    for i in range(1,13):
        col_name = 'luna_' + str(i)
        diferenta_vanzari[col_name] = diferenta_vanzari['dif_comenzi'].shift(i)

    diferenta_vanzari = diferenta_vanzari.dropna().reset_index(drop=True)

    # split into train and test data

    train_data = diferenta_vanzari[:-11]
    test_data = diferenta_vanzari[-11:]

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    x_train, y_train = train_data[:,1:], train_data[:,0:1]
    x_test, y_test = test_data[:,1:], test_data[:,0:1]
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # prediction data frame
    sales_dates = vanzari_lunare['Month'][-12:].reset_index(drop=True)
    predict_df = pd.DataFrame(sales_dates)

    # vanzarile din ultimele 13 luni
    actual_sales = vanzari_lunare['Orders'][-12:].to_list()

    # crearea modelului de linear regression
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    lr_predict = lr_model.predict(x_test)
    lr_predict = lr_predict.reshape(-1, 1)

    lr_predict_test_set  = np.concatenate([lr_predict, x_test], axis=1)
    lr_predict_test_set = scaler.inverse_transform(lr_predict_test_set)

    result_list = []

    for index in range(0, len(lr_predict_test_set)):
        result_list.append(lr_predict_test_set[index][0] + actual_sales[index])

    lr_predict_series = pd.Series(result_list, name="Linear Prediction")
    predict_df = predict_df.merge(lr_predict_series, left_index=True, right_index=True)

    return render(request, 'prediction.html')
