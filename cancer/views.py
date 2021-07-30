from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def index(request):
    return render(request,'index.html')
def about(request):
    return render(request,'about.html')
def bladder(request):
    if request.method=='POST':
        age=request.POST['age']
        bmi=request.POST['bmi']
        glucose=request.POST['glucose']
        insulin=request.POST['insulin']
        homa=request.POST['homa']
        leptin=request.POST['leptin']
        adip=request.POST['adip']
        resis=request.POST['resis']
        mcp=request.POST['mcp']
        sample_data = [age,bmi,glucose,insulin,homa,leptin,adip,resis,mcp]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        data = pd.read_csv('dataR2.csv')
        X = data.drop('Classification',axis=1)
        Y=pd.DataFrame(data['Classification'])
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
        logistic_regression = LogisticRegression(multi_class='ovr')
        logistic_regression.fit(X_train,Y_train)
        a=logistic_regression.score(X_test,Y_test)
        example = pd.DataFrame([age,bmi,glucose,insulin,homa,leptin,adip,resis,mcp])
        example = example.transpose()
        pred=logistic_regression.predict(example)[0]
        if pred==1:
            return render(request,'output1.html')
        else:
            return render(request,'output11.html',{'b':'You Dont have Bladder Cancer.','c':a})
    return render(request,'bladder.html')
def breastcancer(request):
    if request.method == 'POST':
        age = request.POST['age']
        gender = request.POST['gender']
        diagnosis = request.POST['diagnosis']
        tumor = request.POST['tumor']
        lymph = request.POST['lymph']
        insito = request.POST['insito']
        histologic = request.POST['histologic']
        sample_data = [age,gender,diagnosis,tumor,lymph,insito,histologic]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        data = pd.read_csv('ubc_train_dataset.csv')
        X = data.drop('Stage',axis=1)
        Y=pd.DataFrame(data['Stage'])
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
        knn = KNeighborsClassifier()
        knn.fit(X_train,Y_train)
        a=knn.score(X_test,Y_test)
        example = pd.DataFrame([age,gender,diagnosis,tumor,lymph,insito,histologic])
        example = example.transpose()
        pred=knn.predict(example)[0]
        if pred == 1:
            return render(request,'output2.html')
        else:
            return render(request,'output21.html',{'b':'You Dont have Breast Cancer.','c':a})
    return render(request,'breastcancer.html')
def causes(request):
    return render(request,'causes.html')
def cervical(request):
    if request.method == 'POST':
        age = request.POST['age']
        sexual = request.POST['sexual']
        firstsex = request.POST['firstsex']
        preg = request.POST['preg']
        smokes = request.POST['smokes']
        sample_data = [age,sexual,firstsex,preg]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        data = pd.read_csv('risk_factors_cervical_cancer.csv')
        X = data.drop('Biopsy',axis=1)
        Y=pd.DataFrame(data['Biopsy'])
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train,Y_train)
        a=dtree.score(X_test,Y_test)
        example = pd.DataFrame([age,sexual,firstsex,preg,smokes])
        example = example.transpose()
        pred=dtree.predict(example)[0]
        if pred == 1:
            return render(request,'output3.html')
        else:
            return render(request,'output31.html',{'b':'Yo Dont have Cervical Cancer.','c':a})
    return render(request,'cervical.html')
def includes(request):
    return render(request,'includes.html')
def insurence(request):
    if request.method == 'POST':
        age = request.POST['age']
        sex = request.POST['sex']
        bmi = request.POST['bmi']
        child = request.POST['child']
        smoker = request.POST['smoker']
        income = request.POST['income']
        sample_data = [age,sex,bmi,child,smoker,income]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        data = pd.read_csv('insure.csv')
        X = data.drop('claim',axis=1)
        Y=pd.DataFrame(data['claim'])
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=5)
        logistic_regression = LogisticRegression(multi_class='ovr')
        logistic_regression.fit(X_train,Y_train)
        a=logistic_regression.score(X_test,Y_test)
        example = pd.DataFrame([age,sex,bmi,child,smoker,income])
        example = example.transpose()
        pred=logistic_regression.predict(example)[0]
        if pred == 1:
            return render(request,'output.html')
        else:
            return render(request,'output01.html',{'b':'Individual is Not Eligible For Insaurance.','c':a})
    return render(request,'insurence.html')
def op_bladder(request):
    return render(request,'op_bladder.html')
def output(request):
    return render(request,'output.html')
def output1(request):
    return render(request,'output1.html')
def output01(request):
    return render(request,'output01.html')
def output2(request):
    return render(request,'output2.html')
def output3(request):
    return render(request,'output3.html')
def output11(request):
    return render(request,'output11.html')
def output21(request):
    return render(request,'output21.html')
def output31(request):
    return render(request,'output31.html')
def prediction(request):
    return render(request,'prediction.html')