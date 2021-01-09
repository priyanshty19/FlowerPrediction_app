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
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier   
from sklearn.ensemble import RandomForestClassifier

# @st.cache(suppress_st_warning=True)

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
st.sidebar.title("Prediction App")
st.sidebar.header("Navigation Pane")

nav=st.sidebar.radio("",["Home","Data Visualisation","Models","Prediction"])
# col=st.sidebar.selectbox("Choose prediction model",["Logistic Regression","KNN","SVM","Decision Tree","Random Forest"])


if nav=="Home":
    st.image(["data//vesicolor.jpg.png","data//verginica.jpg.png","data//vesicolor.jpg.png"],width=200,height=200)
    

    st.write("## Description of Predictor App")
    st.write("### For the given values of features which are the Sepal Length, Sepal Width,Petal Length,Petal Width and by using different regression models the app can predict the flower.")
    st.write("#### The dataset used is Iris Flower Dataset ")
    st.write("#### It can be downloaded from  https://www.kaggle.com/uciml/iris")
    st.header("Description of DataSet")
    st.subheader("Context")
    st.write("""The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
    This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines""")
    st.subheader("Content")
    st.write("""The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).""")

    # st.subheader("Data Tabulated")

    # if st.checkbox("Show Table"):
    #     st.subheader("Species representation")
    #     dic={
    #         "1":["Iris-setosa"],
    #         "2":["Iris-versicolor"],
    #         "3":["Iris-virginica"]
    #     }
    #     st.table(dic)
    #     st.subheader("DataSet")
    #     st.table(data)



st.set_option('deprecation.showPyplotGlobalUse', False)


if nav=="Data Visualisation":
    st.sidebar.write("# Choose From the following")
    st.header("Visualisation")
    st.write("### Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data ")
    if st.sidebar.checkbox("Count Plot"):
        st.write("## Count Plot")
        sns.countplot(x="species",data=data)
        st.pyplot()
        # st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if st.sidebar.checkbox("Pie Chart"):
        st.write("## Pie Chart")
        data["species"].value_counts().plot.pie(explode=[0.01,0.01,0.01],autopct='%1.1f%%',figsize=(10,8))
        st.pyplot()
    
    if st.sidebar.checkbox("Histogram"):
        st.write("## Histogram")
        data.hist(edgecolor='black',figsize=(10,8))
        st.pyplot()
    
    if st.sidebar.checkbox("Pair Plot"):
        st.write("## Pair Plot")
        sns.pairplot(data,hue="species")
        st.pyplot()
    
    if st.sidebar.checkbox("Heat Plot"):
        st.write("## Heat Plot")
        Correlation_matrix=data.corr().round(2)
        sns.heatmap(data=Correlation_matrix,annot=True)
        st.pyplot()


if nav=="Prediction":
    st.sidebar.write("# Choose From the following")
    col=st.sidebar.selectbox("",["Logistic Regression","KNN","SVM","Decision Tree","Random Forest"])
    
    if col=="Logistic Regression":
        st.header("Know the Flower")
        val1=st.slider("Sepal Length",min_value=0.1,max_value=10.1,step=0.1)
        val2=st.slider("Sepal Width",min_value=0.1,max_value=10.1,step=0.1)
        val3=st.slider("Petal Length",min_value=0.1,max_value=10.1,step=0.1)
        val4=st.slider("Petal Width",min_value=0.1,max_value=10.1,step=0.1)
        val=[[val1,val2,val3,val4]]
        logmodel=LogisticRegression(random_state=101)
        logmodel.fit(x_train,y_train)
        prediction=logmodel.predict(val)

        if st.button("Predict"):
            st.success(f"Predicted Flower is {(prediction)}")
    
    if col=="KNN":
        st.header("Know the Flower")
        val1=st.slider("Sepal Length",min_value=0.1,max_value=10.1,step=0.1)
        val2=st.slider("Sepal Width",min_value=0.1,max_value=10.1,step=0.1)
        val3=st.slider("Petal Length",min_value=0.1,max_value=10.1,step=0.1)
        val4=st.slider("Petal Width",min_value=0.1,max_value=10.1,step=0.1)
        val=[[val1,val2,val3,val4]]
        knn=KNeighborsClassifier(n_neighbors=4,p=2,metric='minkowski',algorithm='auto')
        knn.fit(x_train,y_train)
        prediction2=knn.predict(val)

        if st.button("Predict"):
            st.success(f"Predicted Flower is {(prediction2)}")

    if col=="SVM":
        st.header("Know the Flower")
        val1=st.slider("Sepal Length",min_value=0.1,max_value=10.1,step=0.1)
        val2=st.slider("Sepal Width",min_value=0.1,max_value=10.1,step=0.1)
        val3=st.slider("Petal Length",min_value=0.1,max_value=10.1,step=0.1)
        val4=st.slider("Petal Width",min_value=0.1,max_value=10.1,step=0.1)
        val=[[val1,val2,val3,val4]]
        svm=SVC(kernel='rbf',random_state=0,gamma=0.05,C=1.0)
        svm.fit(x_train,y_train)
        prediction3=svm.predict(val)

        if st.button("Predict"):
            st.success(f"Predicted Flower is {(prediction3)}")

    if col=="Decision Tree":
        st.header("Know the Flower")
        val1=st.slider("Sepal Length",min_value=0.1,max_value=10.1,step=0.1)
        val2=st.slider("Sepal Width",min_value=0.1,max_value=10.1,step=0.1)
        val3=st.slider("Petal Length",min_value=0.1,max_value=10.1,step=0.1)
        val4=st.slider("Petal Width",min_value=0.1,max_value=10.1,step=0.1)
        val=[[val1,val2,val3,val4]]
        tree=DecisionTreeClassifier(random_state=101)
        tree.fit(x_train,y_train)
        prediction4=tree.predict(val)


        if st.button("Predict"):
            st.success(f"Predicted Flower is {(prediction4)}")

    if col=="Random Forest":
        st.header("Know the Flower")
        val1=st.slider("Sepal Length",min_value=0.1,max_value=10.1,step=0.1)
        val2=st.slider("Sepal Width",min_value=0.1,max_value=10.1,step=0.1)
        val3=st.slider("Petal Length",min_value=0.1,max_value=10.1,step=0.1)
        val4=st.slider("Petal Width",min_value=0.1,max_value=10.1,step=0.1)
        val=[[val1,val2,val3,val4]]
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(x_train,y_train)
        prediction5=clf.predict(val)


        if st.button("Predict"):
            st.success(f"Predicted Flower is {(prediction5)}")
    
    # if col=="Average Prediction of Models":
    #     prediction6=(prediction+prediction2+prediction3+prediction4+prediction5)/4
    #     if st.button("Predict"):
    #         st.success(f"Predicted Flower is {(prediction6)}")




if nav=="Models":
    st.header("Models")
    st.subheader("Logistic Regression")
    st.write("### Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).")
   
    logmodel=LogisticRegression(random_state=101)
    logmodel.fit(x_train,y_train)
    prediction=logmodel.predict(x_test)
    st.write(classification_report(y_test,prediction))
    st.write(accuracy_score(y_test,prediction))
    st.write(confusion_matrix(y_test,prediction))

    st.subheader("KNN")
    st.write("### In statistics, the k-nearest neighbors algorithm is a non-parametric machine learning method.It is used for classification and regression. In both cases, the input consists of the k closest training examples in feature space")
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
    st.write("### In machine learning, support-vector machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems.")
    from sklearn.svm import SVC
    svm=SVC(kernel='rbf',random_state=0,gamma=0.05,C=1.0)
    svm.fit(x_train,y_train)
    prediction3=svm.predict(x_test)
    st.write(classification_report(y_test,prediction3))
    st.write(accuracy_score(y_test,prediction3))
    st.write(confusion_matrix(y_test,prediction3))

    st.subheader("Decision Tree")
    st.write("### A decision tree is a flowchart-like structure in which each internal node represents a ""test" "on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).")
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=101)
    tree.fit(x_train,y_train)
    prediction4=tree.predict(x_test)
    st.write(classification_report(y_test,prediction4))
    st.write(accuracy_score(y_test,prediction4))
    st.write(confusion_matrix(y_test,prediction4))

    st.subheader("Random Forest")
    st.write("### Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean/average prediction of the individual trees")
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    prediction5=clf.predict(x_test)
    st.write(classification_report(y_test,prediction5))
    st.write(accuracy_score(y_test,prediction5))
    st.write(confusion_matrix(y_test,prediction5))

    



