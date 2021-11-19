import pandas as pd
import numpy as np
import re
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
        
        # self.bayes_model = make_pipeline(TfidfVectorizer(binary=True),MultinomialNB())
        # self.bayes_model.fit(self.X_train,self.Y_train)
    
        # print("\nBayes Model trained succesfully...")

    def GetBayesModelHeatMap(self):
        labels =self.bayes_model.predict(self.X_test)
        mat = confusion_matrix(self.Y_test,labels)
        sns.heatmap(mat.T)
    
    def GetForestModelHeatMap(self):
        labels =self.forest_model.predict(self.vectorizer.transform(self.X_test).toarray())
        mat = confusion_matrix(self.Y_test,labels)
        sns.heatmap(mat.T, annot=True, fmt='d')
        

    def BayesPredictCategory(self,s):
        pred = self.bayes_model.predict([s])
        print("\nBayes Prediction: "+str(pred[0]))
        print("\tDisaster Tweet") if pred[0] == 1 else print("\tNon Disaster Tweet")

    def ForestPredictCategory(self,s):
        pred = self.forest_model.predict(self.vectorizer.transform([s]).toarray())
        print(pred)
        print("\nForest Prediction: "+str(pred[0]))
        print("\tDisaster Tweet") if pred[0] == 1 else print("\tNon Disaster Tweet")
    
    


cs = Classifier()
#cs.BayesPredictCategory("Fire")
#cs.ForestPredictCategory("Fire")
cs.GetForestModelHeatMap()
plt.xlabel("Predicto")
plt.ylabel("Truth")
plt.savefig('Forest_Predicted.png')
plt.show()