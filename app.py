from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
import string
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
warnings.filterwarnings("ignore")
app = Flask(__name__)

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

def checkProductSentiment(productList,productsSentiments):
  productPercent = {}
  for id in productList:
    filteredProduct = productsSentiments[productsSentiments['productId']==id]
    percentPositive = filteredProduct['Prediction'].sum()/len(filteredProduct)
    productPercent[id]=percentPositive
  productPercentAsc =sorted(productPercent.items(), key=lambda x: x[1])
  finalprodList = [i [0] for i in productPercentAsc[::-1][:5]]
  return finalprodList

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

item_final_ratings = pd.read_pickle('./dataset/item_final_rating')
products = pd.read_csv('dataset/sample30.csv')
products = cleanDataset(products)    
productsSentiments = transformAndPredict(products)
productMapping = products.drop_duplicates(subset='productId',keep='first')[['productId','name','categories']]

def getRecommendedProduct(username,productMapping):
  tablesAlreadyBought = pd.DataFrame()
  df_final = pd.DataFrame()
  df_final5 = pd.DataFrame()
  if username not in item_final_ratings.index:
    userdetails = 'Data Not Available'
  else:
    df_final = item_final_ratings.loc[username].sort_values(ascending=False)[0:20]
    df_final = pd.merge(df_final,productMapping,left_on='productId',right_on='productId',how = 'left')
    productList = list(df_final['productId'])
    final5 = checkProductSentiment(productList,productsSentiments)
    df_final5= df_final[df_final['productId'].isin(final5)]
    tablesAlreadyBought = products[products['userId']==username][['productId','name','categories']]
    userdetails = 'Name - '+ username
  return render_template('home.html',userdetails=userdetails,tablesAlreadyBought=[tablesAlreadyBought.to_html(classes='prediction')],tablesAllReco=[df_final.to_html(classes='prediction')],tables=[df_final5.to_html(classes='prediction')], titles = [])

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return getRecommendedProduct(user,productMapping)

@app.route('/search/names',methods=['GET'])
def process():
    query = request.args.get('query')
    suggestions = list(products[(~products.userId.isnull()) & (products.userId.str.startswith(query))]['userId'])
    suggestions = [{'value':suggestion,'data':suggestion} for suggestion in suggestions]
    return jsonify({"suggestions":suggestions[:5]})

if __name__ == '__main__' :
    app.run(debug=True,use_reloader=True)






