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
    # Bigramas y sus respectivas frecuencias
    country_data = {'Country':['USA','Washington DC','Nigeria','India','WorldWide','Mumbai','UK','New York', 'Cánada'],
                   'Frequency':[100,32,28,24,23,21,20,19,19]}
    country_df = pd.DataFrame(data=country_data).sort_values(by="Frequency",ascending=False)
    # Trigramas y sus respectivas frecuencias
    word_data = {'Word':['suicide bombing','wreckage','derailment','typhoon','oil spill',
                               'outbreak','debris','rescuers'],
                     'Frequency':[64,39,37,37,36,32,32,32]}
    word_df = pd.DataFrame(data=word_data).sort_values(by="Frequency",ascending=False)
    # Datos modelo de prediccion
    pie_data = {'Case':['Succesful','Failed','No Answer'],
                'Percentage':[56.7,33.3,10.0]}
    pie_df = pd.DataFrame(data=pie_data).sort_values(by="Percentage",ascending=False)

    return country_df,word_df,pie_df

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
    word_map = WordCloud(background_color='white',max_words=length,mask=wc_mask,).generate(" ".join(arr))
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
        #header('range')
        sl = slide_bar('',50,200)
        sl.set()
        st.markdown('Refresh data take a few seconds to load the result, so please hold...')
        @st.cache(persist=True,suppress_st_warning=True)
        def swc(df, l):
            return generate_word_cloud(df, l)
        wc = swc(text, sl.value)
        fig = plt.figure(figsize=(8,8))
        plt.imshow(wc,interpolation="bilinear")
        plt.axis('off')
        plt.title('',fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)


    countryDF,wordDF,pieDF = CreateDataFrames()
    #SetPageConfiguration()
    # left_col,right_col = st.columns(2)
    # # Graficas de barras
    # with left_col: 
    salt()

    title("Disaster tweets statistics",40,'gray')
    salt()
    ShowBarGraph(countryDF,"Country","Country with most disaster tweets")
    #with right_col: 
    ShowBarGraph(wordDF,"Word","Most common words in keywords")
    # Grafica de pie
    
    title("Model results",40,'gray')
    # l_col,center_col,r_col = st.columns([0.5,5,0.5])
    # with center_col:
    ShowPieGraph(pieDF,"Model Success")
    