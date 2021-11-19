import pandas as pd
import streamlit as st
import plotly.express as px
import re
from PIL import Image,ImageOps
from wordcloud import WordCloud,STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

class Classifier():
    def __init__(self):
        self.ReadData()
        self.TrainModel()

    def ReadData(self):
        self.data = pd.read_csv("cleanned_data.csv")
        self.df_non_emergency = self.data[self.data['target']==0]
        self.df_emergency = self.data[self.data['target']==1]
        self.non_emergency_X, = self.df_non_emergency.text.sample(3271).fillna(' '),
        self.non_emergency_Y = self.df_non_emergency.target.sample(3271).fillna(int(0))
        self.emergency_X = self.df_emergency.text.fillna(' ')
        self.emergency_Y = self.df_emergency.target.fillna(int(0))
        print("\nData Read succesfully...")


    def TrainModel(self):
        non_emergency_X = self.non_emergency_X.to_numpy()
        emergency_X = self.emergency_X.to_numpy()
        non_emergency_Y = self.non_emergency_Y.to_numpy()
        emergency_Y = self.emergency_Y.to_numpy()
        # Separacion de conjunto de entrenamiento y prueba
        self.X_train = np.concatenate((non_emergency_X[:int(len(non_emergency_X)*0.8)],
                                       emergency_X[:int(len(emergency_X)*0.8)]))
        self.X_test = np.concatenate((non_emergency_X[int(len(non_emergency_X)*0.7):],
                                      emergency_X[int(len(emergency_X)*0.7):]))
        self.Y_train = np.concatenate((non_emergency_Y[:int(len(non_emergency_Y)*0.8)],
                                       emergency_Y[:int(len(emergency_Y)*0.8)]))
        self.Y_test = np.concatenate((non_emergency_Y[int(len(non_emergency_Y)*0.7):],
                                      emergency_Y[int(len(emergency_Y)*0.7):]))
        print("\nData prepared")

        self.vectorizer = CountVectorizer(min_df=1)

        self.forest_model = RandomForestClassifier()
        self.forest_model.fit(self.vectorizer.fit_transform(self.X_train).toarray(), self.Y_train)

        print("\nForest Model trained succesfully...")

        self.bayes_model = make_pipeline(TfidfVectorizer(binary=True),MultinomialNB())
        self.bayes_model.fit(self.X_train,self.Y_train)

        print("\nBayes Model trained succesfully...")

    def GetBayesModelHeatMap(self):
        labels =self.bayes_model.predict(self.X_test)
        mat = confusion_matrix(self.Y_test,labels)
        sns.heatmap(mat.T)

    def GetForestModelHeatMap(self):
        labels =self.forest_model.predict(self.vectorizer.transform(self.X_test).toarray())
        mat = confusion_matrix(self.Y_test,labels)
        sns.heatmap(mat.T)

    def BayesPredictCategory(self,s):
        pred = self.bayes_model.predict([s])
        return "Disaster Tweet" if pred[0] == 1 else "Non Disaster Tweet"

    def ForestPredictCategory(self,s):
        pred = self.forest_model.predict(self.vectorizer.transform([s]).toarray())
        return "Disaster Tweet" if pred[0] == 1 else "Non Disaster Tweet"


class slide_bar:
    value=4
    def __init__(self,title,x,y):
        self.title = title
        self.x=x
        self.y=y
        self.slide_bar = None
        

    def set(self):
        self.slide_bar = st.slider(self.title,self.x,self.y)
        slide_bar.value=self.slide_bar

def CreateDataFrames():
    # frecuencias de paises con mas tweets
    country_data = {'Country':['USA','Washington DC','Nigeria','India','WorldWide','Mumbai','UK','New York', 'Cánada'],
                   'Frequency':[100,32,28,24,23,21,20,19,19]}
    country_df = pd.DataFrame(data=country_data).sort_values(by="Frequency",ascending=False)
    # frecuencias de palabras que más aparecen en el texto de tweets de desastres
    word_data = {'Word':['suicide bombing','wreckage','derailment','typhoon','oil spill',
                               'outbreak','debris','rescuers'],
                     'Frequency':[64,39,37,37,36,32,32,32]}
    word_df = pd.DataFrame(data=word_data).sort_values(by="Frequency",ascending=False)
    # resultados de con los datos de pruebas Random Forest classifier
    test_data = {'Test Data':['predicted truth','predicted false'],
                     'Frequency':[1515,449]}
    test_df = pd.DataFrame(data=test_data).sort_values(by="Frequency",ascending=False)

    # resultados de con los datos de pruebas Naive Bayes
    test_data_nb = {'Test Data':['predicted truth','predicted false'],
                     'Frequency':[1568,396]}
    test_df_nb = pd.DataFrame(data=test_data_nb).sort_values(by="Frequency",ascending=False)

    # Datos modelo de prediccion Random classifier
    pie_data = {'Case':['Succesful','Failed','No Answer'],
                'Percentage':[61.7,32.3,6.0]}
    pie_df = pd.DataFrame(data=pie_data).sort_values(by="Percentage",ascending=False)

    # Datos modelo de prediccion naive bayes
    pie_data_nb = {'Case':['Succesful','Failed','No Answer'],
                'Percentage':[63.1,30.7,6.2]}
    pie_df_nb = pd.DataFrame(data=pie_data_nb).sort_values(by="Percentage",ascending=False)

    # tweets de desastres
    count_tweets = {'Count':[3271,4342],
                     'Target Tweets':['Disaster tweet','Normal tweet']}
    count_tw = pd.DataFrame(data=count_tweets).sort_values(by="Target Tweets",ascending=False)

    return country_df,word_df,test_df,test_df_nb,pie_df,pie_df_nb,count_tw

def SetPageConfiguration():
    st.set_page_config(page_title="Text Prediction",
                       page_icon=":bar_chart:",
                       layout="wide")
    st_style = """
               <style>
               #MainMenu {visibility: hidden;}
               footer {visibility: hidden;}
               header {visibility: hidden;}
               </style>
               """
    st.markdown(st_style,unsafe_allow_html=True)

def SetHeader():
    st.markdown("<h1 style='text-align: center; color: grey; font-size: 100px;'>Disaster Dashboard</h1><br>", unsafe_allow_html=True)

def ShowBarGraph(df,x_label,graph_title):
    bar_graph = px.bar(
        df,
        x=x_label,
        y="Frequency",
        orientation="v",
        color="Frequency",
        color_continuous_scale=["#1446C4","#00C18C","#2E8EC2"],
        height=500,
    )
    bar_graph.update_layout(
        font_color="white",
        title={'text':"<b>"+graph_title+"</b>",
               'x':0.5,
               'xanchor':'center'},
        font_size=20
    )
    
    bar_graph.update_layout({
        'plot_bgcolor': '#ABABAB',
        'paper_bgcolor': '#ABABAB',
        })
    
    st.plotly_chart(bar_graph)

def ShowHorizontalBarGraph(df,x_label,graph_title):
    bar_graph = px.bar(
        df,
        y=x_label,
        x="Count",
        orientation="h",
        color="Count",
        color_continuous_scale=["#1446C4","#00C18C","#2E8EC2"],
        height=350,
    )
    bar_graph.update_layout(
        font_color="white",
        title={'text':"<b>"+graph_title+"</b>",
               'x':0.5,
               'xanchor':'center'},
        font_size=20
    )
    
    bar_graph.update_layout({
        'plot_bgcolor': '#ABABAB',
        'paper_bgcolor': '#ABABAB',
        })
    
    st.plotly_chart(bar_graph)

def ShowPieGraph(df,graph_title):
    pie_graph = px.pie(
        df,
        values="Percentage",
        names="Case",
        title="<b>"+graph_title+"</b>",
        color_discrete_sequence=["#1446C4","#00C18C","#2E8EC2"],
        height=650,
    )
    pie_graph.update_layout(
        title={'text':"<b>"+graph_title+"</b>",
               'x':0.5,
               'xanchor':'center'},
        font_size=24
    )
    st.plotly_chart(pie_graph)

def generate_word_cloud(arr, length):
    wc_mask = Image.open('images/fire.jpg')
    wc_mask = ImageOps.grayscale(wc_mask)
    wc_mask = np.array(wc_mask)
    word_map = WordCloud(background_color='white',mask=wc_mask,).generate(" ".join(arr))
    return word_map

def title(text,size,color):
    st.markdown(f'<h1 style="font-weight:bolder;font-size:{size}px;color:{color};text-align:center;">{text}</h1>',unsafe_allow_html=True)

def header(text):
    st.markdown(f"<p style='color:gray;'>{text}</p>",unsafe_allow_html=True)

def salt():
    st.markdown(f"<br><br>",unsafe_allow_html=True)

def test_function(text):
    return text

def modelInput(function,model_name):
    title('Predict with {} model'.format(model_name),20,'#1446C4')
    tweet = st.text_input(f"Enter a tweet", key=model_name)
    result = st.button(f"Predict", key=model_name)
    prediction = ""

    if result:
        prediction = function(tweet)


    st.markdown(f"<br> Tweet: {tweet}", unsafe_allow_html=True)
    st.markdown(f"Prediction: {prediction}", unsafe_allow_html=True)

if __name__ == "__main__":

    data_tw = pd.read_csv('clean_data.csv')

    keywords = data_tw.loc[data_tw['keyword'] != 'None']
    keywords = data_tw.loc[data_tw['keyword'] != 'none']
    tweets = [words.split(',') for words in keywords['keyword']]
    tweets = [y for x in tweets for y in x]
    text = [words.split(',') for words in keywords['text']]
    text = [y for x in text for y in x]
    SetHeader()

    title("WordCloud of a disaster tweets",40,'gray')
    l_col,center_col,r_col = st.columns([0.5,5,0.5])
    
    with center_col:
        @st.cache(persist=True,suppress_st_warning=True)
        def swc(df, l):
            return generate_word_cloud(df, l)
        wc = swc(text, 200)
        fig = plt.figure(figsize=(8,8))
        plt.imshow(wc,interpolation="bilinear")
        plt.axis('off')
        plt.title('',fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)


    countryDF,wordDF,testDF,testDFnb,pieDF,pieDFnb,countTW = CreateDataFrames()
    
    salt()
    title("Disaster tweets statistics",40,'gray')
    salt() 
    ShowHorizontalBarGraph(countTW,"Target Tweets","Target Tweets")
    ShowBarGraph(countryDF,"Country","Country with most disaster tweets")
    
    ShowBarGraph(wordDF,"Word","Most common words in keywords")
    # Grafica de pie
    salt()
    title("Random Forest Classifier Model results",40,'gray')
    salt()
    ShowBarGraph(testDF,"Test Data","Random Forest Classifier prediction using test data")
    
    # with center_col:
    ShowPieGraph(pieDF,"Model Success")
    
    title("Naive Bayes Model results",40,'gray')
    salt()

    ShowBarGraph(testDFnb,"Test Data","Naive Bayes prediction using test data")
    
    ShowPieGraph(pieDFnb,"Model Success")

    @st.cache(persist=True,suppress_st_warning=True)
    def makeClassifier():
        return Classifier()

    cs = makeClassifier()

    modelInput(cs.BayesPredictCategory, 'Naive Bayes')
    modelInput(cs.ForestPredictCategory, 'Random Forest')
    