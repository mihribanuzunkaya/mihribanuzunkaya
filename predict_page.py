from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle
import numpy as np
import lime
import lime.lime_tabular
from streamlit import components
st.set_option('deprecation.showPyplotGlobalUse', False)

s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',')
s_all.drop(['Participant'],axis=1, inplace= True)
X=s_all.drop(['Class'] , axis=1) 
y=s_all['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

def load_model():
    with open ('knn_allselected.pkl', 'rb') as file :
        data = pickle.load(file)
    return data    

data = load_model()

knn_grid = data ["knn"]

def load_model():
    with open ('mlp_allselected.pkl', 'rb') as file :
        data = pickle.load(file)
    return data    

data = load_model()

mlp_grid = data ["mlp"]

def load_model():
    with open ('svm_allselected.pkl', 'rb') as file :
        data = pickle.load(file)
    return data    

data = load_model()

svm_grid = data ["svm"]

def load_model():
    with open ('rf_allselected.pkl', 'rb') as file :
        data = pickle.load(file)
    return data    

data = load_model()

rfc_grid = data ["rf"]

def load_model():
    with open ('dt_allselected.pkl', 'rb') as file :
        data = pickle.load(file)
    return data    

data = load_model()

dtree_grid = data ["dt"]

def show_predict_page():
    st.title("XAI Based Autism Spectrum Disorder Diagnosis Web Interfece")

    st.write ("""### Please fill the given features below""")
    
    a18 = st.number_input("A18 = Action Net Dwell Time [ms]",step=0.01,format="%.2f")
    a19 = st.number_input("A19 = Action Dwell Time [ms]",step=0.01,format="%.2f" )
    a20 = st.number_input("A20 = Action Glance Duration [ms]",step=0.01,format="%.2f" )
    a21 = st.number_input("A21 = Action Diversion Duration [ms]",step=0.01,format="%.2f" )
    a25 = st.number_input("A25 = Action Fixation Count",step=0.01,format="%.2f" )
    a26 = st.number_input("A26 = Action Net Dwell Time [%]",step=0.01,format="%.2f" )
    a27 = st.number_input("A27 = Action Dwell Time [%]",step=0.01,format="%.2f" )
    a28 = st.number_input("A28 = Action Fixation Time [ms]",step=0.01,format="%.2f" )
    a29 = st.number_input("A29 = Action Fixation Time [%]",step=0.01,format="%.2f" )
    f16 = st.number_input("F16 = Face Entry Time [ms]",step=0.01,format="%.2f" )
    f17 = st.number_input("F17 = Face Sequence",step=0.01,format="%.2f" )
    f18 = st.number_input("F18 = Face Net Dwell Time [ms]",step=0.01,format="%.2f" )
    f19 = st.number_input("F19 = Face Dwell Time [ms]",step=0.01,format="%.2f" )
    f20 = st.number_input("F20 = Face Glance Duration [ms]",step=0.01,format="%.2f" )
    f21 = st.number_input("F21 = Face Diversion Duration [ms]",step=0.01,format="%.2f" )
    f22 = st.number_input("F22 = Face First Fixation Duration [ms]",step=0.01,format="%.2f" )
    f23 = st.number_input("F23 = Face Glances Count",step=0.01,format="%.2f" )
    f24 = st.number_input("F24 = Face Revisits",step=0.01,format="%.2f" )
    f25 = st.number_input("F25 = Face Fixation Count",step=0.01,format="%.2f" )
    f26 = st.number_input("F26 = Face Net Dwell Time [%]",step=0.01,format="%.2f" )
    f27 = st.number_input("F27 = Face Dwell Time [%]",step=0.01,format="%.2f" )
    f28 = st.number_input("F28 = Face Fixation Time [ms][ms]",step=0.01,format="%.2f" )
    f29 = st.number_input("F29 = Face Fixation Time [%]",step=0.01,format="%.2f" )
    f30 = st.number_input("F30 = Average Fixation Duration [ms]",step=0.01,format="%.2f" )
    FixationCount = st.number_input("Event Fixation Time",step=0.01,format="%.2f")
    FixationDurationAverage = st.number_input("Event Fixation Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationAverage = st.number_input("Event Saccade Duration Average [ms]",step=0.01,format="%.2f" )
    SaccadeDurationMaximum = st.number_input("Event Saccade Duration Maximum",step=0.01,format="%.2f" )
    SaccadeAmplitudeMinimum = st.number_input("Event Saccade Amplitude Minimum",step=0.01,format="%.2f" )
    SaccadeLatencyAverage = st.number_input("Event Saccade Latency Average [ms]",step=0.01,format="%.2f" )
    BlinkDurationTotal = st.number_input("Event Blink Duration Total [ms]",step=0.01,format="%.2f")

    classifier_name = st.selectbox("Select Classifer",("KNN", "SVM", "MLP","Random Forest","Decision Tree"))
    ok = st.button("Make Prediction")

    if ok :

        
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)

        if classifier_name == "KNN":
            prediction = knn_grid.predict(X)
            prediction_proba = knn_grid.predict_proba(X)
            st.subheader("The KNN Prediction is")
            st.write(prediction)
            st.subheader("KNN Prediction Probability")
            st.write(prediction_proba)

        elif classifier_name == "SVM" :
            prediction = svm_grid.predict(X)
            st.subheader("The SVM Prediction is")
            st.write(prediction)
            predict_proba = svm_grid.predict_proba(X)

        elif classifier_name == "Random Forest":
            prediction = rfc_grid.predict (X)
            prediction_proba = rfc_grid.predict_proba(X) 
            st.subheader("The Random Forest Prediction is")
            st.write(prediction)
            st.subheader("Random Forest Prediction Probability")
            st.write(prediction_proba)

        elif classifier_name == "MLP":
            prediction = mlp_grid.predict(X)  
            prediction_proba = mlp_grid.predict_proba(X) 
            st.subheader("The MLP Prediction is")
            st.write(prediction)
            st.subheader("MLP Prediction Probability")
            st.write(prediction_proba)  

        else :
            prediction = dtree_grid.predict(X)
            prediction_proba = dtree_grid.predict_proba(X) 
            st.subheader("The Decision Tree Prediction is")
            st.write(prediction)
            st.subheader("Decision Tree Prediction Probability")
            st.write(prediction_proba)
                
    if st.checkbox("LIME Explanation") :
        X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
        X = X.astype(float)
        if classifier_name == "KNN":
            
            df = s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
            

            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
            lime_exp= lime_explainer.explain_instance(data_row = np.array(feature_list), predict_fn = knn_grid.predict_proba, num_features=33)

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)

        elif classifier_name == "SVM":
            
            st.write("SVM predict_proba=false iken LIME'ı desteklemiyor.")
        elif classifier_name == "MLP":
            
            df = s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
            
            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
            lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = mlp_grid.predict_proba, num_features=33, )

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)
        elif classifier_name=="Random Forest":
            

            df = s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
            
            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
            lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = rfc_grid.predict_proba, num_features=33, )

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)
    
        else :
            df = s_all = pd.read_csv(r"C:\Users\Mihriban\Desktop\notebook\3_xAll_AFB_Event_selected.csv",delimiter=',') 
            X = np.array([[a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]]) 
            X = X.astype(float)
            feature_list = [a18 ,a19,a20,a21,a25,a26,a27,a28,a29,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,FixationCount,FixationDurationAverage,SaccadeDurationAverage,SaccadeDurationMaximum,SaccadeAmplitudeMinimum,SaccadeLatencyAverage,BlinkDurationTotal]

            feature_names = ['a18' ,'a19','a20','a21','a25','a26','a27','a28','a29','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','FixationCount','FixationDurationAverage','SaccadeDurationAverage','SaccadeDurationMaximum','SaccadeAmplitudeMinimum','SaccadeLatencyAverage','BlinkDurationTotal']
            class_names = ['NG', 'OSB']

            import lime
            from lime import lime_tabular
            
            lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values, feature_names=feature_names, class_names=class_names, mode = 'classification', verbose = True, random_state=0)
            lime_exp= lime_explainer.explain_instance(np.array(feature_list), predict_fn = dtree_grid.predict_proba, num_features=33, )

            html_lime = lime_exp.as_html()
            components.v1.html(html_lime, width=1100, height=350, scrolling=True)



            
            
            

        

            








    
        





