from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
from model import ModelConfig
warnings.filterwarnings("ignore")
app = Flask(__name__)

products = ModelConfig().getProducts()
productsSentiments = ModelConfig().getProductsSentiments()
productMapping = ModelConfig().getProductMapping()
final_ratings = ModelConfig().get_final_ratings()

def checkProductSentiment(productList,productsSentiments):
  productPercent = {}
  for id in productList:
    filteredProduct = productsSentiments[productsSentiments['productId']==id]
    percentPositive = filteredProduct['Prediction'].sum()/len(filteredProduct)
    productPercent[id]=percentPositive
  productPercentAsc =sorted(productPercent.items(), key=lambda x: x[1])
  finalprodList = [i [0] for i in productPercentAsc[::-1][:5]]
  return finalprodList

def getRecommendedProduct(username,productMapping):
  tablesAlreadyBought = pd.DataFrame()
  df_final = pd.DataFrame()
  df_final5 = pd.DataFrame()
  if username not in final_ratings.index:
    userdetails = 'Data Not Available'
  else:
    df_final = final_ratings.loc[username].sort_values(ascending=False)[0:20]
    df_final = pd.merge(df_final,productMapping,left_on='productId',right_on='productId',how = 'left')
    productList = list(df_final['productId'])
    final5 = checkProductSentiment(productList,productsSentiments)
    df_final5= df_final[df_final['productId'].isin(final5)]
    tablesAlreadyBought = products[products['userId']==username][['productId','name','categories']]
    userdetails = 'Name - '+ username
  return render_template('index.html',userdetails=userdetails,tablesAlreadyBought=[tablesAlreadyBought.to_html(classes='prediction')],tablesAllReco=[df_final.to_html(classes='prediction')],tables=[df_final5.to_html(classes='prediction')], titles = [])

@app.route('/')
def index():
    return render_template('index.html')


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


if __name__ == "__main__":
  app.run()





