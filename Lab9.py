import pandas as pd
import streamlit as st
import plotly.express as px

def CreateDataFrames():
    # Bigramas y sus respectivas frecuencias
    bigram_data = {'Bigram':['dont,know','cant,wait','last,year','new,york','last,night','high,school','feel,like','years,ago'],
                   'Frequency':[1341,1337,1320,1320,1078,947,940,930]}
    bigram_df = pd.DataFrame(data=bigram_data).sort_values(by="Frequency",ascending=False)
    # Trigramas y sus respectivas frecuencias
    trigram_data = {'Trigram':['cant,wait,see','happy,mothers,days','let,us,know','new,york,city',
                               'happy,new,year','two,years,ago','new,york,times','dont,even,know'],
                     'Frequency':[246,235,213,171,142,110,93,91]}
    trigram_df = pd.DataFrame(data=trigram_data).sort_values(by="Frequency",ascending=False)
    # Datos modelo de prediccion
    pie_data = {'Case':['Succesful','Failed','No Answer'],
                'Percentage':[56.7,33.3,10.0]}
    pie_df = pd.DataFrame(data=pie_data).sort_values(by="Percentage",ascending=False)

    return bigram_df,trigram_df,pie_df

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
    st.markdown("<h1 style='text-align: center; color: white; font-size: 100px;'>Text Prediction</h1>", unsafe_allow_html=True)

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
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
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


if __name__ == "__main__":
    bigramDF,trigramDF,pieDF = CreateDataFrames()
    SetPageConfiguration()
    SetHeader()
    left_col,right_col = st.columns(2)
    # Graficas de barras
    with left_col: 
        ShowBarGraph(bigramDF,"Bigram","Most Common Bigrams")
    with right_col: 
        ShowBarGraph(trigramDF,"Trigram","Most Common Trigrams")
    # Grafica de pie
    st.markdown("##")
    l_col,center_col,r_col = st.columns(3)
    with center_col:
        ShowPieGraph(pieDF,"Model Precision")
    