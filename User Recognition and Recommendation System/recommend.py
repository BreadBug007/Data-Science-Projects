import json
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')


def get_config(config_file='config.ini'):
  import configparser
  config = configparser.ConfigParser()
  config.read(config_file)
  return config


def train_model(model_name="word2vec.model",
                sales_data="sales_data/OnlineRetail.csv",
                products_json="products.json"):
  import pandas as pd
  import random
  from tqdm import tqdm

  '''
    reading data
  '''
  df = pd.read_csv(sales_data)
  # removing all null rows
  df.dropna(inplace=True)
  '''
     data preparation
  '''
  # converting stockcode to str type
  df['StockCode'] = df['StockCode'].astype(str)

  '''
    splitting data
  '''
  customers = df["CustomerID"].unique().tolist()

  # shuffle customer ID's
  random.shuffle(customers)

  # extract 90% of customer ID's
  customers_train = [customers[i] for i in range(round(0.9 * len(customers)))]

  # split data into train and validation set
  train_df = df[df['CustomerID'].isin(customers_train)]
  validation_df = df[~df['CustomerID'].isin(customers_train)]

  # create sequences of purchases made by the customers in the dataset for both the train and validation set
  # list to capture purchase history of the customers
  purchases_train = []

  # populate the list with the product codes
  for i in tqdm(customers_train, desc='Creating sequences'):
    temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_train.append(temp)

  # list to capture purchase history of the customers
  purchases_val = []

  # populate the list with the product codes
  for i in tqdm(validation_df['CustomerID'].unique(),
                desc='Creating sequences'):
    temp = validation_df[validation_df["CustomerID"]
                         == i]["StockCode"].tolist()
    purchases_val.append(temp)

  '''
    train word2vec model
  '''
  print("Training word2vec model")
  model = Word2Vec(min_count=1,
                   sg=1,  # using skip-gram so 1
                   hs=0,  # using negative sampling
                   negative=5,  # for negative sampling
                   alpha=0.03,
                   min_alpha=0.0007,
                   seed=14,
                   window=10)

  model.build_vocab(purchases_train,
                    progress_per=200)

  model.train(purchases_train,
              total_examples=model.corpus_count,
              epochs=10,
              report_delay=1)
  model.init_sims(replace=True)
  print("Trained word2vec model:")
  print(model)

  # save word2vec model
  model.save(model_name)

  '''
    initialize and save a product dict mapping product id to description
  '''
  products = train_df[["StockCode", "Description"]]

  # remove duplicates
  products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

  # create product-ID and product-description dictionary
  products_dict = products.groupby(
      'StockCode')['Description'].apply(list).to_dict()
  with open(products_json, 'w') as f:
    json.dump(products_dict, f)


def get_recommendations(stock_code,
                        n=6,
                        model_name='word2vec.model',
                        products_json='products.json'):
  model = Word2Vec.load(model_name)
  with open(products_json, 'r') as f:
    products_dict = json.load(f)

  # extract most similar products for the input vector
  v = model[stock_code]
  ms = model.similar_by_vector(v, topn=n + 1)[1:]

  # extract name and similarity score of the similar products
  new_ms = []
  for j in ms:
    pair = (products_dict[j[0]][0], j[1])
    new_ms.append(pair)
  return new_ms


def connect_to_database(config_file):
  import mysql.connector
  mydb = mysql.connector.connect(
      host=config_file['mysql']['host'],
      port=config_file['mysql']['port'],
      user=config_file['mysql']['user'],
      passwd=config_file['mysql']['pass'],
      database=config_file['mysql']['db'],
  )
  return mydb


def retrieve_products_mysql(user_name, config_file):
  mydb = connect_to_database(config_file)
  mycursor = mydb.cursor()
  query = f"SELECT product FROM `sales` WHERE user = '{user_name}' LIMIT 10"
  mycursor.execute(query)
  records = mycursor.fetchall()
  products = []
  for record in records:
    products.append(record[0])
  mydb.close()
  return products


def retrieve_products_firebase(user_name, config_file):
  import requests

  project = config_file['firebase']['project']
  table = config_file['firebase']['table']
  url = f"https://{project}.firebaseio.com/{table}.json"
  resp = requests.get(url)
  json_data = resp.json()
  products = []
  for entry in json_data:
    if entry['user'] == user_name:
      products.append(entry['product'])
  return products


def email_alert(msg_text, config_file):
  import smtplib
  from email.mime.multipart import MIMEMultipart
  from email.mime.text import MIMEText
  from datetime import datetime

  username = config_file['email']['sender_user']
  password = config_file['email']['sender_pass']
  # If you want to always cc to a mail for testing, add that email below
  recepient_addrs = []
  recepient_addrs.append(config_file['email']['recipient'])
  file_list = []

  def send_mail(send_from: str, subject: str, html: str,
                send_to: list, files=None):

    send_to = send_to

    msg = MIMEMultipart('alternative')
    msg['From'] = send_from
    msg['To'] = ', '.join(send_to)
    msg['Subject'] = subject

    msg.attach(MIMEText(html, 'html'))

    smtp = smtplib.SMTP(host="smtp.gmail.com", port=587)
    smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

  send_mail(send_from=username,
            subject="Product Recommendations",
            html="""\
              <html>
                <head></head>
                <body>
                  <h3>Top product recommendations (%s)</h3>
                  <ol>%s</ol>
                </body>
              </html>
            """ % (datetime.today().strftime('%Y-%m-%d'),
                   msg_text.replace("\n", "<br>")),
            send_to=recepient_addrs,
            files=file_list
            )


def recommend_products(user_name, n=5):
  # get configuration defined in config.ini file
  config_file = get_config()

  # get data from mysql or firebase
  # product_codes = retrieve_products_mysql(user_name, config_file)
  product_codes = retrieve_products_firebase(user_name, config_file)

  # get recommendations
  all_recommendations = []
  for product_code in product_codes:
    recommendations = get_recommendations(product_code)
    all_recommendations.extend(recommendations)
  all_recommendations.sort(key=lambda x: x[1], reverse=True)
  recommendation_text = ""
  print(f"Top {n} recommendations are:")
  for i in all_recommendations[:n]:
    print(f"\"{i[0]}\" with a similarity score of {i[1]}")
    recommendation_text += f"<li>{i[0]}</li>"

  # send an email with the recommendations
  print("\nSending an email with the recommendations...")
  email_alert(recommendation_text, config_file)
