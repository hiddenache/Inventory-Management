from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import (
    View,
    CreateView, 
    UpdateView
)
import os
import json
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib import messages
from .models import Stock
from .forms import StockForm
from django_filters.views import FilterView
from .filters import StockFilter
import csv
from django.http import HttpResponse
from transactions.models import PurchaseBill
from django.utils import timezone
from django.db.models.functions import ExtractMonth
from django.db.models import Count
import calendar
from django.db.models import Sum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from django.shortcuts import render

class StockListView(FilterView):
    filterset_class = StockFilter
    queryset = Stock.objects.filter(is_deleted=False)
    template_name = 'inventory.html'
    paginate_by = 10

class StockCreateView(SuccessMessageMixin, CreateView):                                 
    model = Stock                                                                       
    form_class = StockForm                                                              
    template_name = "edit_stock.html"                                                 
    success_url = '/inventory'                                                          
    success_message = "Stoc creat cu succes"                             

    def get_context_data(self, **kwargs):                                               
        context = super().get_context_data(**kwargs)
        context["title"] = 'Adauga produs'
        context["savebtn"] = 'Adauga'
        return context       

class StockUpdateView(SuccessMessageMixin, UpdateView):                                 
    model = Stock                                                                       
    form_class = StockForm                                                              
    template_name = "edit_stock.html"                                                   
    success_url = '/inventory'                                                          
    success_message = "Produsul a fost modificat cu succes"                             

    def get_context_data(self, **kwargs):                                               
        context = super().get_context_data(**kwargs)
        context["title"] = 'Editeaza produs'
        context["savebtn"] = 'Update Produs'
        context["delbtn"] = 'Sterge produs'
        return context

class StockDeleteView(View):                                                            
    template_name = "delete_stock.html"                                                 
    success_message = "Produs sters cu succes!"                             
    
    def get(self, request, pk):
        stock = get_object_or_404(Stock, pk=pk)
        return render(request, self.template_name, {'object' : stock})

    def post(self, request, pk):  
        stock = get_object_or_404(Stock, pk=pk)
        stock.is_deleted = True
        stock.save()                                               
        messages.success(request, self.success_message)
        return redirect('inventory')
    
class ExportDataExcel(View):
    def export_data_to_excel(request):
        
        data = PurchaseBill.objects.all()

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="vanzari.csv"'
        writer = csv.writer(response)
  
        headers = ["Month", "Orders"]
        writer.writerow(headers)
        
        for purchase_bill in data:
            month = purchase_bill.time.date()
            orders = 1  
            row = [month, orders]
            writer.writerow(row)

        return response

class PredictSales(View):
    template_name = 'prediction.html'
    def get(self, request):
        return render(request, 'prediction.html')
    
    def post(self, request):
        file = request.FILES.get('file')  
        
        if not file:
            error = 'Please select a file.'
            return render(request, self.template_name, {'error': error})
        
        vanzari = pd.read_csv(file)  
        vanzari['Month'] = pd.to_datetime(vanzari['Month'])
        vanzari['Month'] = vanzari['Month'].dt.to_period('M')

        vanzari_lunare = vanzari.groupby('Month')['Orders'].sum().reset_index()
        
        vanzari_lunare['Month'] = vanzari_lunare['Month'].dt.to_timestamp()
        vanzari_lunare['dif_comenzi'] = vanzari_lunare['Orders'].diff()
        vanzari_lunare = vanzari_lunare.dropna()

        diferenta_vanzari = vanzari_lunare.drop(['Month', 'Orders'], axis=1)

        for i in range(1,13):
            col_name = 'luna_' + str(i)
            diferenta_vanzari[col_name] = diferenta_vanzari['dif_comenzi'].shift(i)

        diferenta_vanzari = diferenta_vanzari.dropna().reset_index(drop=True)

        train_data = diferenta_vanzari[:-12]
        test_data = diferenta_vanzari[-12:]

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        x_train, y_train = train_data[:,1:], train_data[:,0:1]
        x_test, y_test = test_data[:,1:], test_data[:,0:1]
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        sales_dates = vanzari_lunare['Month'][-12:].reset_index(drop=True)
        predict_df = pd.DataFrame(sales_dates)

        actual_sales = vanzari_lunare['Orders'][-13:].to_list()
                
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

        lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], vanzari_lunare['Orders'][-12:]))
        lr_mae = mean_absolute_error(predict_df['Linear Prediction'], vanzari_lunare['Orders'][-12:])
        lr_r2 = r2_score(predict_df['Linear Prediction'], vanzari_lunare['Orders'][-12:])
        
        prediction_data = {
            'labels': predict_df['Month'].astype(str).tolist(),
            'actualSales': actual_sales,
            'predictedSales': predict_df['Linear Prediction'].tolist()
        }
        prediction_data_json = json.dumps(prediction_data)

        return render(request, 'prediction.html', {
            'lr_mse': lr_mse,
            'lr_mae': lr_mae,
            'lr_r2': lr_r2,
            'prediction_data_json': prediction_data_json
        })
    

        