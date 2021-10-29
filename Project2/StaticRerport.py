import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image,ImageOps
from wordcloud import WordCloud,STOPWORDS
import numpy as np
import matplotlib.pyplot as plt

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
    # resultados de con los datos de pruebas bert classifier
    test_data = {'Test Data':['predicted truth','predicted false'],
                     'Frequency':[1515,449]}
    test_df = pd.DataFrame(data=test_data).sort_values(by="Frequency",ascending=False)

    # resultados de con los datos de pruebas Naive Bayes
    test_data_nb = {'Test Data':['predicted truth','predicted false'],
                     'Frequency':[1568,396]}
    test_df_nb = pd.DataFrame(data=test_data_nb).sort_values(by="Frequency",ascending=False)

    # Datos modelo de prediccion bert classifier
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
    title("Bert Classifier Model results",40,'gray')
    salt()
    ShowBarGraph(testDF,"Test Data","Bert Classifier prediction using test data")
    
    # with center_col:
    ShowPieGraph(pieDF,"Model Success")
    
    title("Naive Bayes Model results",40,'gray')
    salt()

    ShowBarGraph(testDFnb,"Test Data","Naive Bayes prediction using test data")
    
    ShowPieGraph(pieDFnb,"Model Success")
    