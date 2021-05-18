import pandas as pd
import re
import string
import pickle
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def concatTitleText(title,text):
    if pd.isnull(title):
        return text
    else:
        return title + '. ' + text

def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and ( not w.isdigit() and len(w)>3) )]) 
    text3 = re.sub(r'[^\w\s]','',text2)
    return text3.lower()

def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

def transformAndPredict(df):
    df['review'] = df.apply(lambda x: concatTitleText(x['reviews_title'],x['reviews_text']),axis=1)
    df['review'] = df['review'].apply(clean_text)
    df['review']=  df['review'].apply(remove_stopwords)
    X = df['review']
    tfidfconverter = TfidfVectorizer(max_features=4178, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    tfidfconverter.fit(X.tolist())
    pipeline = pickle.load(open('pickle/logreg_scenario_1.pkl', 'rb'))
    vecs=tfidfconverter.transform(df['reviews_text'].tolist())
    pred=pipeline.predict(vecs)
    prediction = pipeline.predict_proba(vecs)
    df_final = pd.DataFrame()
    df_final['productId'] = df['productId']
    df_final['name'] = df['name']
    df_final['reviews_text'] = df['reviews_text']
    df_pred_prob = pd.DataFrame(prediction, columns=['Negative','Positive'])
    df_final['max_prob'] = df_pred_prob[['Negative','Positive']].max(axis=1)
    df_final['max_prob_class'] = df_pred_prob.idxmax(axis=1)
    df_final['Prediction'] = pred
    return df_final

def cleanDataset(products):
    products.rename(columns = {'id' : 'productId'},inplace=True)
    products.rename(columns = {'reviews_username' : 'userId'},inplace=True)
    products.rename(columns = {'reviews_rating' : 'rating'},inplace=True)
    products = products[~(products['userId'].isnull())]
    products = products[~(products['userId']=='byamazon customer')]
    products['combine'] = products['productId']+products['userId']
    products.drop_duplicates(subset='combine',keep='first',inplace=True)
    products.drop(['combine'],axis=1,inplace=True)
    return products

class ModelConfig:
    def __init__(self):
        """Initialize instance of ModelConfig."""
        self.products = cleanDataset(pd.read_csv('./dataset/sample30.csv'))
        self.productsSentiments = transformAndPredict(self.products)
        self.user_final_rating = pd.read_pickle('./pickle/user_final_rating',compression='zip')
        self.productMapping  = self.products.drop_duplicates(subset='productId',keep='first')[['productId','name','categories']]
    
    def getProducts(self):
        return self.products

    def getProductsSentiments(self):
        return self.productsSentiments
    
    def getProductMapping(self):
        return self.productMapping

    def get_final_ratings(self):
        return self.user_final_rating
