from django.shortcuts import render
from django.views.generic import View, TemplateView
from inventory.models import Stock
from django.db.models.functions import ExtractMonth
from django.db.models import Count
from transactions.models import PurchaseBillDetails, SaleBill, PurchaseBill
import calendar
from collections import defaultdict

class HomeView(View):
    template_name = "home.html"
    def get(self, request):        
        labels = []
        data = []
        month_orders = []
        month_names = []
        stockqueryset = Stock.objects.filter(is_deleted=False).order_by('-quantity')
        months = PurchaseBill.objects.annotate(month=ExtractMonth('time')).values('month').annotate(count=Count('billno')).order_by('-count')
        for item in stockqueryset:
            labels.append(item.name)
            data.append(item.quantity)
        for month in months:
            month_names.append(calendar.month_name[month['month']])
            month_orders.append(month['count'])
            
        for order in PurchaseBill.objects.all():
            month = order.time.month
        sales = SaleBill.objects.order_by('-time')[:3]
        purchases = PurchaseBill.objects.order_by('-time')[:3]
        context = {
            'labels'    : labels,
            'data'      : data,
            'sales'     : sales,
            'purchases' : purchases,
            'month_orders': month_orders,
            'month_names'   : month_names
        }
        return render(request, self.template_name, context)