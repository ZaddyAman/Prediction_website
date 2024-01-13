import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from flask import Flask, request, render_template
import pandas as pd 

import joblib


app = Flask(__name__)

model_albumin = joblib.load('model_albumin.pkl')
model_alanine = joblib.load('model_alanine.pkl')
model_ast = joblib.load('model_ast.pkl')
model_phosphotase = joblib.load('model_phosphotase.pkl')
model_nitrogen = joblib.load('model_nitrogen.pkl')
model_calcium = joblib.load('model_calcium.pkl')
model_cholesterol = joblib.load('model_cholesterol.pkl')
model_bicarbonate = joblib.load('model_bicarbonate.pkl')
model_iron = joblib.load('model_iron.pkl')
model_phosphorus = joblib.load('model_phosphorus.pkl')
model_bilirubin = joblib.load('model_bilirubin.pkl')
model_protein = joblib.load('model_protein.pkl')
model_Creatinine = joblib.load('model_Creatinine.pkl')
model_sodium = joblib.load('model_sodium.pkl')
model_pottasium = joblib.load('model_pottasium.pkl')
model_chloride = joblib.load('model_chloride.pkl')
model_globulin = joblib.load('model_globulin.pkl')
model_glucose = joblib.load('model_glucose.pkl')


@app.route('/') 
def home():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    out = np.array(input_data)
    input_1 = [[out[0],out[18]]]
    input_2 = [[out[1],out[18]]]
    input_3 = [[out[0],out[18]]]
    input_4 = [[out[1],out[18]]]
    input_5 = [[out[0],out[18]]]
    input_6 = [[out[1],out[18]]]
    input_7 = [[out[0],out[18]]]
    input_8 = [[out[1],out[18]]]
    input_9 = [[out[0],out[18]]]
    input_10 = [[out[1],out[18]]]
    input_11 = [[out[0],out[18]]]
    input_12 = [[out[1],out[18]]]
    input_13 = [[out[0],out[18]]]
    input_14 = [[out[1],out[18]]]
    input_15 = [[out[0],out[18]]]
    input_16 = [[out[1],out[18]]]
    input_17 = [[out[0],out[18]]]
    input_18 = [[out[1],out[18]]]

    results = [] 
    
    
    if out[0] != -1:
        df1 = pd.DataFrame(input_1, columns=['albumin', 'gender'])
        if not df1.empty:
            result_albumin = model_albumin.predict(df1)
            results.append(result_albumin)
    else:
        
        results.append("No prediction for model_albumin")
        
    
    
    if out[1] != -1:
        df1 = pd.DataFrame(input_1, columns=['  alanine', 'gender'])
        if not df1.empty:
            result_alanine = model_alanine.predict(df1)
            results.append(result_alanine)
    else:
        
        results.append("No prediction for model_alanine")        
    
    
    
    if out[2] != -1:
        df1 = pd.DataFrame(input_1, columns=['ast', 'gender'])
        if not df1.empty:
            result_ast = model_ast.predict(df1)
            results.append(result_ast)
    else:
        
        results.append("No prediction for model_ast")
        
    
    
    if out[3] != -1:
        df1 = pd.DataFrame(input_1, columns=['phosphotase', 'gender'])
        if not df1.empty:
            result_phosphotase = model_phosphotase.predict(df1)
            results.append(result_phosphotase)
    else:
        
        results.append("No prediction for model_phosphotase")
    
    
    
    if out[4] != -1:
        df1 = pd.DataFrame(input_1, columns=[' nitrogen', 'gender'])
        if not df1.empty:
            result_nitrogen = model_nitrogen.predict(df1)
            results.append(result_nitrogen)
    else:
        
        results.append("No prediction for model_nitrogen")
        
    
    
    if out[5] != -1:
        df1 = pd.DataFrame(input_1, columns=['calcium', 'gender'])
        if not df1.empty:
            result_calcium = model_calcium.predict(df1)
            results.append(result_calcium)
    else:
        
        results.append("No prediction for model_calcium")        
    
    
    
    if out[6] != -1:
        df1 = pd.DataFrame(input_1, columns=['cholesterol', 'gender'])
        if not df1.empty:
            result_cholesterol = model_cholesterol.predict(df1)
            results.append(result_cholesterol)
    else:
        
        results.append("No prediction for model_cholesterol")
        
    
    
    if out[7] != -1:
        df1 = pd.DataFrame(input_1, columns=['bicarbonate', 'gender'])
        if not df1.empty:
            result_bicarbonate = model_bicarbonate.predict(df1)
            results.append(result_bicarbonate)
    else:
        
        results.append("No prediction for model_bicarbonate")
         
    
    
    if out[8] != -1:
        df1 = pd.DataFrame(input_1, columns=['iron', 'gender'])
        if not df1.empty:
            result_iron = model_iron.predict(df1)
            results.append(result_iron)
    else:
        
        results.append("No prediction for model_iron")
        
    
    
    if out[9] != -1:
        df1 = pd.DataFrame(input_1, columns=['phosphorus', 'gender'])
        if not df1.empty:
            result_phosphorus = model_phosphorus.predict(df1)
            results.append(result_phosphorus)
    else:
        
        results.append("No prediction for model_phosphorus")        
    
    
    
    if out[10] != -1:
        df1 = pd.DataFrame(input_1, columns=['bilirubin', 'gender'])
        if not df1.empty:
            result_bilirubin = model_bilirubin.predict(df1)
            results.append(result_bilirubin)
    else:
        
        results.append("No prediction for model_bilirubin")
        
    
    
    if out[11] != -1:
        df1 = pd.DataFrame(input_1, columns=['protein', 'gender'])
        if not df1.empty:
            result_protein = model_protein.predict(df1)
            results.append(result_protein)
    else:
        
        results.append("No prediction for model_protein")
    
    
    
    if out[12] != -1:
        df1 = pd.DataFrame(input_1, columns=['Creatinine ', 'gender'])
        if not df1.empty:
            result_Creatinine = model_Creatinine.predict(df1)
            results.append(result_Creatinine)
    else:
        
        results.append("No prediction for model_Creatinine")
        
    
    
    if out[13] != -1:
        df1 = pd.DataFrame(input_1, columns=['sodium', 'gender'])
        if not df1.empty:
            result_sodium = model_sodium.predict(df1)
            results.append(result_sodium)
    else:
        
        results.append("No prediction for model_sodium")        
    
    
    
    if out[14] != -1:
        df1 = pd.DataFrame(input_1, columns=['pottasium', 'gender'])
        if not df1.empty:
            result_pottasium = model_pottasium.predict(df1)
            results.append(result_pottasium)
    else:
        
        results.append("No prediction for model_pottasium")
        
    
    
    if out[15] != -1:
        df1 = pd.DataFrame(input_1, columns=['chloride', 'gender'])
        if not df1.empty:
            result_chloride = model_chloride.predict(df1)
            results.append(result_chloride)
    else:
        
        results.append("No prediction for model_chloride")
        
        
    if out[16] != -1:
        df1 = pd.DataFrame(input_1, columns=['globulin ', 'gender'])
        if not df1.empty:
            result_globulin = model_globulin.predict(df1)
            results.append(result_globulin)
    else:
        
        results.append("No prediction for model_globulin")
    
    
    if out[17] != -1:
        df1 = pd.DataFrame(input_1, columns=['glucose', 'gender'])
        if not df1.empty:
            result_glucose = model_glucose.predict(df1)
            results.append(result_glucose)
    else:
        
        results.append("No prediction for model_glucose")    
        
    
          
         
   

        
        
        
    
    return render_template('index.html', predict='output {}'.format(results))    




if __name__ == "__main__":
  app.run(port=8000, debug='on')


