from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.StockListView.as_view(), name='inventory'),
    path('prediction', views.PredictSales.as_view(), name='predict'),
    path('new', views.StockCreateView.as_view(), name='new-stock'),
    path('export', views.ExportDataExcel.export_data_to_excel, name='export-data'),
    path('stock/<pk>/edit', views.StockUpdateView.as_view(), name='edit-stock'),
    path('stock/<pk>/delete', views.StockDeleteView.as_view(), name='delete-stock'),
]