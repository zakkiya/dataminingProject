import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
from pickle import dump
import joblib
import altair as alt

st.sidebar.title('CEK !!!!')
dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Data", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    url = "https://www.kaggle.com/datasets/prathamtripathi/drug-classification"
    st.markdown(
        f'[Dataset]({url})')
    st.write('Obat yang cocok sesuai...')

    dt = pd.read_csv('https://raw.githubusercontent.com/zakkiya/dataminingProject/master/drug200%20(1).csv')
    st.dataframe(dt)
    with preporcessing:
        preporcessing, ket = st.tabs(['preporcessing', 'Ket preporcessing'])
        with ket:
            st.write("""
                    Keterangan:
                    * 0 : Tidak 
                    * 1 : Iya
                    """)
        with preporcessing:
            X = dt.drop('Drug',axis=1)
            y = dt['Drug']
            age = dt[["Age"]]
            X

            # Normalisasi MinMax
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(age)
            features_names = age.columns.copy()
            scaled_features = pd.DataFrame(scaled,columns=features_names)
            dt["Age"] = scaled
            dt

            # Splitting Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,random_state=1)
    with modeling:
        # x = dt.drop('Drug',axis=1)
        # y = dt[['Drug']].values
        # # split data
        # X_train, X_test, y_train, y_test = train_test_split(
        # X, y, train_size=0.5,random_state=1)

        X = dt.drop('Drug',axis=1)
        y = dt['Drug']
        age = dt[["Age"]]
        X
        # Normalisasi MinMax
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(age)
        features_names = age.columns.copy() 
        scaled_features = pd.DataFrame(scaled,columns=features_names)
        dt["Age"] = scaled
        dt

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(y)

        ybaru = le.transform(y)
        ybaru
        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X, ybaru, train_size=0.5,random_state=1)

        mlpc, knc, dtc = st.tabs(
        ["MLPClassifier", "KNeighborsClassifier", "DecisionTreeClassifier"])
        with mlpc:
            clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            y_pred_clf = clf.predict(X_test)
            akurasi_clf = accuracy_score(y_test, y_pred_clf)
            label_clf = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_clf}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_clf}')
            st.dataframe(label_clf)
        with knc:
            knn = KNeighborsClassifier(n_neighbors = 5)
            knn.fit(X_train,y_train)
            y_pred_knn = knn.predict(X_test)
            akurasi_knn = accuracy_score(y_test, y_pred_knn)
            label_knn = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_knn}')
            st.dataframe(label_knn)
        with dtc:
            classifier=DecisionTreeClassifier(criterion='gini')
            classifier.fit(X_train,y_train)
            y_pred_d3 = classifier.predict(X_test)
            akurasi_d3 = accuracy_score(y_test, y_pred_d3)
            label_d3 = pd.DataFrame(
            data={'Label Test': y_test, 'Label Predict': y_pred_d3}).reset_index()
            st.success(f'akurasi terhadap data test = {akurasi_d3}')
            st.dataframe(label_d3)
    with implementation:
         data = pd.read_csv('https://raw.githubusercontent.com/zakkiya/dataminingProject/master/drug200%20(1).csv')

         x = data.drop(columns='Drug')
         x

         y = data['Drug'].values
         y

         #normalisasi

        # tema = st.selectbox('Sex', ['F', 'M'])
        # temp_F = 1 if tema == 'F' else 0
        # temp_M = 1 if tema == 'M' else 0

        # outlook = st.selectbox('BP', ['HIGH', 'LOW', 'NORMAL'])
        # outlook_HIGH = 1 if outlook == 'HIGH' else 0
        # outlook_LOW = 1 if outlook == 'LOW' else 0
        # outlook_NORMAL = 1 if outlook == 'NORMAL' else 0

        # humidity = st.selectbox('Cholesterol', ['HIGH', 'NORMAL'])
        # humidity_HIGH = 1 if humidity == 'HIGH' else 0
        # humidity_NORMAL = 1 if humidity == 'NORMAL' else 0

        # # data = np.array([[temp_F,temp_M,outlook_HIGH,outlook_LOW,outlook_NORMAL,humidity_HIGH,humidity_NORMAL]])
        # data = [tema,]
        # model = st.selectbox('Pilih Model', ['MLPC', 'KNN', 'DTREE'])
        # if model == 'MLPC':
        #     y_imp = clf.predict(data)
        # elif model == 'KNN':
        #     y_imp = knn.predict(data)
        # else :
        #     y_imp = classifier.predict(data)
        # st.success(f'Model yang dipilih = {model}')
        # st.success(f'Data Predict = {y_imp}')
