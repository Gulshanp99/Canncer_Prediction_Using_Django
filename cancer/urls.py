from typing import ValuesView
from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('index',views.index,name='index'),
    path('about',views.about,name='about'),
    path('bladder',views.bladder,name='bladder'),
    path('breastcancer',views.breastcancer,name='breastcancer'),
    path('causes',views.causes,name='causes'),
    path('cervical',views.cervical,name='cervical'),
    path('includes',views.includes,name='includes'),
    path('insurence',views.insurence,name='insurence'),
    path('pharma',views.op_bladder,name='op_bladder'),
    path('output',views.output,name='output'),
    path('output1',views.output1,name='output1'),
    path('output01',views.output01,name='output01'),
    path('output2',views.output2,name='output2'),
    path('output3',views.output3,name='output3'),
    path('output11',views.output11,name='output11'),
    path('output21',views.output21,name='output21'),
    path('output31',views.output31,name='output31'),
    path('predict',views.prediction,name='prediction'),
]