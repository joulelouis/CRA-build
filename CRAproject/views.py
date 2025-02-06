# from django.http import HttpResponse
from django.shortcuts import render 

def basepage(request):
    return render(request, 'base.html')

def delta_method(request):
    # return HttpResponse("My Feature_1 Page")
    return render(request, 'delta_method.html')

def homepage(request):
    # return HttpResponse("Hello World! I'm Home")
    return render(request, 'home.html')


def feature_1(request):
    # return HttpResponse("My Feature_1 Page")
    return render(request, 'feature_1.html')

def feature_2(request):
    # return HttpResponse("My Feature_2 Page")
    return render(request, 'feature_2.html')