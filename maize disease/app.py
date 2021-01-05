import math
import numpy as np
import pandas as pd
import pickle

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral3,cividis

from flask import Flask, render_template, request

#data analytics library.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
#random forest
from sklearn.ensemble import RandomForestClassifier

week_df = pd.read_csv('weekdata _new.csv')

months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
import numpy as np
def month_change(mon):
    temp = mon.lower()[0:3]
    if temp in months.keys():
        return months[temp]
week_df["mon"] = week_df["Month"].apply(month_change)
week_df["mon"] = week_df["mon"].fillna(0).astype(np.int64)
week_df_temp = week_df[week_df['mon'] != 0]
year_df = week_df_temp['mon'].apply(str)+ '-' +week_df_temp['Year'].map(str)
final_df = pd.concat([week_df_temp, year_df],axis=1,sort=False)
final_df = final_df.rename(columns={0:"date"})
final_df["datetime"] = pd.to_datetime(final_df["date"],format = "%m-%Y")


palette = ['#ba32a0', '#f85479', '#f8c260', '#00c2ba']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'

def SVM_model(data):
    X = week_df[['Min Temperature', 'Max Temperature ', 'Minimum RH', 'Maximum RH', 'Rainfall']]
    y = week_df['Class Label']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

def random_model(data):
    X = week_df[['Min Temperature', 'Max Temperature ', 'Minimum RH', 'Maximum RH', 'Rainfall']]
    y = week_df['Class Label']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    ra_acc = metrics.accuracy_score(y_test, y_pred)
    return ra_acc


def knn_clasifier_model(data):
    print("inside knn_clasifier_model")

    X = week_df[[ 'Max Temperature ','Min Temperature', 'Maximum RH', 'Minimum RH', 'Rainfall']]
    y = week_df['Class Label']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    #build model
    knn.fit(X_train, y_train)

    #prdict a test model.
    y_pred = knn.predict(X_test)

    #display results
    a = accuracy_score(y_test, y_pred)

    return a

def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None
def all_line_chart(dataset):
    date_time_values = list(dataset["datetime"])
    max_temp_values = list(dataset["Max Temperature "])
    min_temp_values = list(dataset["Min Temperature"])
    max_RH_values = list(dataset["Maximum RH"])
    min_RH_values = list(dataset["Minimum RH"])
    Rain_values = list(dataset["Rainfall"])
    source = ColumnDataSource(data={
        'dt':date_time_values,
        'max_temv':max_temp_values,
        'min_temv':min_temp_values,
        'max_rhv':max_RH_values,
        'min_rhv':min_RH_values,
        'rainv':Rain_values
    })

    hover_tool = HoverTool(
        tooltips=[('time in years', '@dt'), ('maxx_temp', '@max_temv'),('minn_temp', '@min_temv'),('maxx_rh','@max_rhv'),('minn_rh','@min_rhv'),('rain_v','@rainv')]
    )

    p = figure(tools=[hover_tool], plot_height=400,plot_width=500, title='Data Distribution',x_axis_type="datetime")
    p.line(x='dt', y='max_temv', line_width=2, source=source, color=Spectral3[2], legend='Max Temprature')
    p.line(x='dt', y='min_temv', line_width=2, source=source, color='grey', legend='Min Temprature')
    p.line(x='dt', y='max_rhv', line_width=2, source=source, color='yellow', legend='Max RH')
    p.line(x='dt', y='min_rhv', line_width=2, source=source, color='red', legend='Min RH')
    p.line(x='dt', y='rainv', line_width=2, source=source, color='blue', legend='Rainfall')
    
    plot_styler(p)
    
    return p


def temp_line_chart(dataset):
    print("inside temp_line_chart")
    date_time_values = list(dataset["datetime"])
    
    max_temp_values = list(dataset["Max Temperature "])
    
    min_temp_values = list(dataset["Min Temperature"])

    source = ColumnDataSource(data={
        'dt':date_time_values,
        'mt':max_temp_values,
        'mit':min_temp_values
    })

    hover_tool = HoverTool(
        tooltips=[('time in years', '@dt'), ('max_temp', '@mt'),('min_temp', '@mit')]
    )

    p = figure(tools=[hover_tool], plot_height=400, title='temperature graph',x_axis_type="datetime")
    p.line(x='dt', y='mt', line_width=2, source=source, color=Spectral3[2], legend='Max Temperature')
    p.line(x='dt', y='mit', line_width=2, source=source, color='blue', legend='Min Temperature')
    plot_styler(p)
    #p.xaxis.ticker = source.data['dt']
    #p.sizing_mode = 'scale_width'
    return p


def RH_line_chart(dataset):
    print("inside RH_line_chart")
    date_time_values = list(dataset["datetime"])
    max_RH_values = list(dataset["Maximum RH"])
    min_RH_values = list(dataset["Minimum RH"])

    source = ColumnDataSource(data={
        'dt':date_time_values,
        'max_RH':max_RH_values,
        'min_RH':min_RH_values
    })

    hover_tool = HoverTool(
        tooltips=[('time in years', '@dt'), ('max_RH', '@max_RH'),('min_RH', '@min_RH')]
    )

    p = figure(tools=[hover_tool], plot_height=400, title='RH Graph',x_axis_type="datetime")
    p.line(x='dt', y='max_RH', line_width=2, source=source, color=Spectral3[2], legend='Max RH')
    p.line(x='dt', y='min_RH', line_width=2, source=source, color='blue', legend='Min RH')
    plot_styler(p)
    #p.xaxis.ticker = source.data['dt']
    #p.sizing_mode = 'scale_width'
    
    return p

def Rain_line_chart(dataset):
    print("inside Rain_line_chart")
    date_time_values = list(dataset["datetime"])
    Rain_values = list(dataset["Rainfall"])

    source = ColumnDataSource(data={
        'dt':date_time_values,
        'rain':Rain_values,
    })

    hover_tool = HoverTool(
        tooltips=[('time in years', '@dt'), ('rain', '@rain')]
    )

    p = figure(tools=[hover_tool], plot_height=400, title='Rainfall Graph',x_axis_type="datetime")
    #p.box(x='rain', top='dt', width=0.7, source=source, color="blue", legend='Rainfall')
    p.vbar(x='dt', top='rain', source=source, line_color='blue')
    plot_styler(p)
    #p.xaxis.ticker = source.data['dt']
    #p.sizing_mode = 'scale_width'
    
    return p


def redraw(p_class):
    print("inside the redraw")
    print("redraw p_class:",p_class)
    if p_class == 1:
        print("inside redraw 1")
        temp_chart = temp_line_chart(final_df)
        return (temp_chart)
    elif p_class == 2:
        print("inside redraw 2")
        RH_chart = RH_line_chart(final_df)
        return (RH_chart)
    elif p_class == 3:
        print("inside redraw 3")
        Rain_chart = Rain_line_chart(final_df)
        return (Rain_chart)
    elif p_class == 4:
        print("inside redraw 4")
        knn_clasifier_model(final_df)
    else:
        print("inside redraw else")
        temp_chart = temp_line_chart(final_df)
        return (temp_chart)


app = Flask(__name__)

# @app.route('/')
# def chart():
#     return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/chartpage', methods=['GET', 'POST'])
def chart():
    selected_class = request.form.get('dropdown-select')
    print(week_df)
    print("selected class:",selected_class)
    print(type(selected_class))

    if selected_class == '0' or selected_class == None:
        all_chart=all_line_chart(final_df)
        script_all_chart,div_all_chart = components(all_chart)
        return render_template('index.html',selected_class=selected_class,script_all_chart=script_all_chart,div_all_chart=div_all_chart)
        
    elif selected_class == '1':
        print("inside chart condition 1")
        redrawn_chart = redraw(1)
        script_temp_chart,div_temp_chart = components(redrawn_chart)
        return render_template(
                        'index.html',
                        selected_class=selected_class,
                        script_temp_chart= script_temp_chart,
                        div_temp_chart = div_temp_chart)

    elif selected_class == '2':
        print("inside chart condition 2")
        redrawn_RH_chart = redraw(2)
        script_RH_chart,div_RH_chart = components(redrawn_RH_chart)
        return render_template(
                         'index.html',
                         selected_class=selected_class,
                         script_RH_chart=script_RH_chart,
                         div_RH_chart=div_RH_chart)
    elif selected_class == '3':
        print("inside chart condition 3")
        redrawn_Rain_chart = redraw(3)
        script_Rain_chart,div_Rain_chart = components(redrawn_Rain_chart)
        return render_template(
                         'index.html',
                         selected_class=selected_class,
                         script_Rain_chart=script_Rain_chart,
                         div_Rain_chart=div_Rain_chart)
    elif selected_class == '4':
        print("inside chart condition 3")
        redrawn_Rain_chart = redraw(4)
    else:
        print("inside chart the else condition",selected_class)
        redrawn_chart = redraw(1)
        script_temp_chart,div_temp_chart = components(redrawn_chart)
        return render_template(
                        'index.html',
                        selected_class=selected_class,
                        script_temp_chart= script_temp_chart,
                        div_temp_chart = div_temp_chart)

from flask import render_template


@app.route('/models', methods=['GET', 'POST'])
def paddy(name=None):
    selected_model = request.form.get('check')
    print("selected model is",selected_model)
    if selected_model == "1":
        res = {}
        res["accuracy"] = knn_clasifier_model(final_df)
        res["model"] = "KNN Model"
        return render_template('model.html', name=res)
    elif selected_model == "2":
        res = {}
        res["accuracy"] = SVM_model(final_df)
        res["model"] = "SVM Model"
        return render_template('model.html', name=res)
    elif selected_model == "3":
        res = {}
        res["accuracy"] = random_model(final_df)
        res["model"] = "Random Forest Model"
        return render_template('model.html', name=res)
    else:
        return render_template('model.html')

def paddy_predict_diseases(pre_list):
    X = week_df[[ 'Max Temperature ','Min Temperature', 'Maximum RH', 'Minimum RH', 'Rainfall']]
    y = week_df['Class Label']

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)

    to_predict = np.array(pre_list).reshape(1, 5)
    y_pred=clf.predict(to_predict)
    return y_pred
    

@app.route('/predict', methods=['GET', 'POST'])
def predict_diseases():
    to_predict_list = request.form.to_dict()
    print(to_predict_list)
    for key,val in to_predict_list.items():
        if len(val) == 0:
            pre_str = "Please fill all the input fields."
            return render_template('results.html', pre_str = pre_str)
    if to_predict_list:
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        res = paddy_predict_diseases(to_predict_list)
        if res:
            pre_str = "the diseases will spread in above condition"
            return render_template('results.html', pre_str = pre_str)
        else:
            pre_str = "the diseases will not spread in above condition"
            return render_template('results.html', pre_str=pre_str)
    return render_template('results.html')

@app.route('/datasetview',methods=['GET', 'POST'])
def datasetview():
    datalength=len(week_df)
    return render_template('datasetv.html',data=week_df,datalength=datalength)


import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model_resnet50.h5')


def prdict_image(img_path):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="leaf blight:<br><br><br><br><br>information about leaf blight<br><br><br><br><br>preventive measures<br><br><br><br><br>see treatment"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
    
    
    return preds


@app.route('/predictimage', methods=['GET', 'POST'])
def predictimage():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)

        # Make prediction
        preds = prdict_image(file_path)
        result=preds
        return render_template('predictdata.html',data=preds)
    return render_template('predictdata.html')




if __name__ == '__main__':
    app.run(debug=True)