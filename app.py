import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score


data=pd.read_csv("data//IRIS.csv")

x=data.drop("species",axis=1)
y=data["species"] 
x_train ,x_test ,y_train ,y_test=train_test_split(x,y,test_size=.24,random_state=101)

spe={"species":{
    "Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3
}}
data.replace(spe,inplace=True)

page_bg_img = '''
<style>
body {
background-image: url("https://img.freepik.com/free-vector/bright-background-with-dots_1055-3132.jpg?size=338&ext=jpg&ga=GA1.2.1846610357.1604275200");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)




st.title("Flower predictor")

nav=st.sidebar.radio("Navigation",["Home","Data Visualisation","Models","Prediction"])
if nav=="Home":
    st.image("data//Flower.jpg.png",width=800,height=1000)
    
    st.header("Description of DataSet")
    st.subheader("Context")
    st.write("""The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
    This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines""")
    st.subheader("Content")
    st.write("""he dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).""")

    st.subheader("Data Tabulated")
    if st.checkbox("Show Table"):
        st.table(data.head(10))




if nav=="Data Visualisation":
    st.header("Visualisation")
    if st.checkbox("Count Plot"):
        sns.countplot(x="species",data=data)
        st.pyplot()
    
    if st.checkbox("Pie Chart"):
        data["species"].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',figsize=(10,8))
        st.pyplot()
    
    if st.checkbox("Histogram"):
        data.hist(edgecolor='black',figsize=(10,8))
        st.pyplot()
    
    if st.checkbox("Pair Plot"):
        sns.pairplot(data,hue="species")
        st.pyplot()
    
    if st.checkbox("Heat Plot"):
        Correlation_matrix=data.corr().round(2)
        sns.heatmap(data=Correlation_matrix,annot=True)
        st.pyplot()


if nav=="Prediction":
    st.header("Know the Flower")

    val1=st.slider("Sepal Length",min_value=0.1,max_value=20.1,step=0.1)
    val2=st.slider("Sepal Width",min_value=0.1,max_value=20.1,step=0.1)
    val3=st.slider("Petal Length",min_value=0.1,max_value=20.1,step=0.1)
    val4=st.slider("Petal Width",min_value=0.1,max_value=20.1,step=0.1)
    val=[[val1,val2,val3,val4]]
    logmodel=LogisticRegression(random_state=101)
    logmodel.fit(x_train,y_train)
    prediction=logmodel.predict(val)

    if st.button("Predict"):
        st.success(f"Predicted Flower is {(prediction)}")


if nav=="Models":
    st.header("Models")
    
    st.subheader("Logistic Regression")
   
    logmodel=LogisticRegression(random_state=101)
    logmodel.fit(x_train,y_train)
    prediction=logmodel.predict(x_test)
    st.write(classification_report(y_test,prediction))
    st.write(accuracy_score(y_test,prediction))
    st.write(confusion_matrix(y_test,prediction))

    st.subheader("KNN")
    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import KNeighborsClassifier
    nn=NearestNeighbors(5)
    nn.fit(data)
    knn=KNeighborsClassifier(n_neighbors=4,p=2,metric='minkowski',algorithm='auto')
    knn.fit(x_train,y_train)
    prediction2=knn.predict(x_test)
    st.write(classification_report(y_test,prediction2))
    st.write(accuracy_score(y_test,prediction2))
    st.write(confusion_matrix(y_test,prediction2))

    st.subheader("SVM")
    from sklearn.svm import SVC
    svm=SVC(kernel='rbf',random_state=0,gamma=0.05,C=1.0)
    svm.fit(x_train,y_train)
    prediction3=svm.predict(x_test)
    st.write(classification_report(y_test,prediction3))
    st.write(accuracy_score(y_test,prediction3))
    st.write(confusion_matrix(y_test,prediction3))

    st.subheader("Decision Tree")
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=101)
    tree.fit(x_train,y_train)
    prediction4=tree.predict(x_test)
    st.write(classification_report(y_test,prediction4))
    st.write(accuracy_score(y_test,prediction4))
    st.write(confusion_matrix(y_test,prediction4))

    st.subheader("Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    prediction5=clf.predict(x_test)
    st.write(classification_report(y_test,prediction5))
    st.write(accuracy_score(y_test,prediction5))
    st.write(confusion_matrix(y_test,prediction5))

    



