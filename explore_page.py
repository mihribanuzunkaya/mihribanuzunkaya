
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from streamlit_shap import st_shap
import shap 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle 
import seaborn as sns
import streamlit.components.v1 as components

import eli5
from eli5 import formatters
from IPython.display import display

s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',')
s_all.drop(['Participant'],axis=1, inplace= True)
X=s_all.drop(['Class'] , axis=1) 
y=s_all['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

model = RandomForestClassifier()
model.fit(X_train,y_train)



def show_explore_page():
    st.title("Açıklanabilir Yapay Zekaya Dayalı Otizm Teşhisi Web Arayüzü")
    st.sidebar.write ("""### Please fill the given features below""")
    
    a18 = st.sidebar.number_input("A18 = Action Net Dwell Time [ms]",step=0.01,format="%.2f")
    a19 = st.sidebar.number_input("A19 = Action Dwell Time [ms]",step=0.01,format="%.2f" )
    a20 = st.sidebar.number_input("A20 = Action Glance Duration [ms]",step=0.01,format="%.2f" )
    a21 = st.sidebar.number_input("A21 = Action Diversion Duration [ms]",step=0.01,format="%.2f" )
    a25 = st.sidebar.number_input("A25 = Action Fixation Count",step=0.01,format="%.2f" )
    a26 = st.sidebar.number_input("A26 = Action Net Dwell Time [%]",step=0.01,format="%.2f" )
    a27 = st.sidebar.number_input("A27 = Action Dwell Time [%]",step=0.01,format="%.2f" )
    a28 = st.sidebar.number_input("A28 = Action Fixation Time [ms]",step=0.01,format="%.2f" )
    a29 = st.sidebar.number_input("A29 = Action Fixation Time [%]",step=0.01,format="%.2f" )
    f16 = st.sidebar.number_input("F16 = Face Entry Time [ms]",step=0.01,format="%.2f" )
    f17 = st.sidebar.number_input("F17 = Face Sequence",step=0.01,format="%.2f" )
    f18 = st.sidebar.number_input("F18 = Face Net Dwell Time [ms]",step=0.01,format="%.2f" )
    f19 = st.sidebar.number_input("F19 = Face Dwell Time [ms]",step=0.01,format="%.2f" )
    f20 = st.sidebar.number_input("F20 = Face Glance Duration [ms]",step=0.01,format="%.2f" )
    f21 = st.sidebar.number_input("F21 = Face Diversion Duration [ms]",step=0.01,format="%.2f" )
    f22 = st.sidebar.number_input("F22 = Face First Fixation Duration [ms]",step=0.01,format="%.2f" )
    f23 = st.sidebar.number_input("F23 = Face Glances Count",step=0.01,format="%.2f" )
    f24 = st.sidebar.number_input("F24 = Face Revisits",step=0.01,format="%.2f" )
    f25 = st.sidebar.number_input("F25 = Face Fixation Count",step=0.01,format="%.2f" )
    f26 = st.sidebar.number_input("F26 = Face Net Dwell Time [%]",step=0.01,format="%.2f" )
    f27 = st.sidebar.number_input("F27 = Face Dwell Time [%]",step=0.01,format="%.2f" )
    f28 = st.sidebar.number_input("F28 = Face Fixation Time [ms][ms]",step=0.01,format="%.2f" )
    f29 = st.sidebar.number_input("F29 = Face Fixation Time [%]",step=0.01,format="%.2f" )
    f30 = st.sidebar.number_input("F30 = Average Fixation Duration [ms]",step=0.01,format="%.2f" )
    FixationCount = st.sidebar.number_input("Event Fixation Time",step=0.01,format="%.2f")
    FixationDurationAverage = st.sidebar.number_input("Event Fixation Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationAverage = st.sidebar.number_input("Event Saccade Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationMaximum = st.sidebar.number_input("Event Saccade Duration Maximum",step=0.01,format="%.2f" )
    SaccadeAmplitudeMinimum = st.sidebar.number_input("Event Saccade Amplitude Minimum",step=0.01,format="%.2f" )
    SaccadeLatencyAverage = st.sidebar.number_input("Event Saccade Latency Average [ms]",step=0.01,format="%.2f" )
    BlinkDurationTotal = st.sidebar.number_input("Event Blink Duration Total [ms]",step=0.01,format="%.2f")

    

    ok = st.button("See Random Forest Prediction")

    if ok :
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)

        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)
        st.subheader("Random Forest Prediction")
        st.write(prediction)
        st.subheader(" Random Forest Prediction Probability")
        st.write(prediction_proba)

        st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.checkbox("SHAP & ELI5 Explanation") :  
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)  
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)
        
        st.title("SHAP Explanation")
        explainer = shap.TreeExplainer(model)
        shap_values= explainer.shap_values(X_test)
        
        st.header("Test Veri Seti")
        st.write(X_test)
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X_test),400)

        st.pyplot(shap.summary_plot(shap_values[1], X_test))

        feature_names=X_train.columns.values
        class_names = ["NG" , "OSB"]
        html_object = eli5.show_prediction(model,X[0],feature_names=feature_names,target_names = class_names)
        
        raw_html = html_object._repr_html_()
        components.html(raw_html, width=1100, height=350, scrolling=True)

        
        
        
