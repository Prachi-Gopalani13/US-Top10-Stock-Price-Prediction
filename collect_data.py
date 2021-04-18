# Import libraries
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas_datareader import data as pdr
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

import string
from urllib.request import urlopen
from urllib.request import Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad
from sklearn.feature_extraction.text import TfidfVectorizer
from yahoo_fin import stock_info as si
import datetime
import time
import yfinance as yf
import plotly.express as px
yf.pdr_override()
# Parameters 
n = 10 #the # of article headlines displayed per ticker

##################################################################################

index_name = list(['^GSPC']) # S&P 500, Dow Jones, Nasdaq , '^IXIC',,'^DJI'
start_date = datetime.datetime.now() - datetime.timedelta(days=7)
end_date = datetime.date.today()
exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "5 Day MA", "5 day Low", "5 day High"])
returns_multiples = []

#######################Select top 50 companies from 1 year performance ###################################

comp = pd.read_csv('names_1yr.csv')

tickers = list(comp['Stock'])


# Index Returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_df

index_df[['Percent Change_GSPC']] = index_df['Adj Close'].pct_change()
index_df 

index_return_GSPC = (index_df['Percent Change_GSPC'] + 1).cumprod()[-1]
index_return_GSPC

#index_return_DJI = (index_df['Percent Change_DJI'] + 1).cumprod()[-1]
#index_return_DJI

#index_return__IXIC = (index_df['Percent Change_IXIC'] + 1).cumprod()[-1]
#index_return__IXIC

m = {}
pc = {}
sr = {}


########################## companywise ########################################
def calc_return(tickers,index_return,index_name):
      for ticker in tickers:
              # Download historical data as CSV for each stock (makes the process faster)
              df = pdr.get_data_yahoo(ticker, start_date, end_date)
              df.to_csv(f'/Users/himanshi/Desktop/project_Stock/data_week/{ticker}.csv')

              # Calculating returns relative to the market (returns multiple)
              df['Percent Change'] = df['Adj Close'].pct_change()
              stock_return = (df['Percent Change'] + 1).cumprod()[-1]
    
              returns_multiple = round((stock_return / index_return), 2)
              returns_multiples.extend([returns_multiple])
    
              print (f'Ticker: {ticker}; Returns Multiple against {index_name}: {returns_multiple}\n')
              m[ticker] = returns_multiple
              pc[ticker] = stock_return
              sr[ticker] = df['Percent Change']

#calc_return(dow_list, index_return_DJI, "Dow Jones")
#calc_return(nasdaq_list, index_return__IXIC, "NASDAQ")
calc_return(tickers, index_return_GSPC, "S&P500")


# Creating dataframe of only top 10% for the last week
rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
#rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.90)]

rs_df.head(10)  # top 10%


rs_stocks = rs_df['Ticker']

# Checking conditions of top 50 of stocks in given list
for stock in rs_stocks:    
    try:
        df = pd.read_csv(f'/Users/himanshi/Desktop/project_Stock/data_week/{stock}.csv', index_col=0)
        sma = [5]  # simple moving average
        for x in sma:
            df["SMA_"+str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)  # calculating sma 
        
        # Storing required values 
        currentClose = df["Adj Close"][-1]
        moving_average_5 = df["SMA_5"][-1]
        low_of_5days = round(min(df["Low"][-5:]), 2)  # 5 days a week
        high_of_5days = round(max(df["High"][-5:]), 2)
        RS_Rating = round(rs_df[rs_df['Ticker']==stock].RS_Rating.tolist()[0])
        
     

        # Condition 1: Current Price > 5 SMA
        condition_1 = currentClose >   moving_average_5
    
        # Condition 2: Current Price is at least 30% above 5 day low
        condition_2 = currentClose >= (1.1*low_of_5days)
           
        # Condition 3: Current Price is within 25% of 5 day high
        condition_3 = currentClose >= (.6*high_of_5days)
        
        # If all conditions above are true, add stock to exportList
        if(condition_1):
            exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating ,"5 Day MA": moving_average_5, "5 day Low": low_of_5days, "5 day High": high_of_5days}, ignore_index=True)
            print (stock + " made the Minervini requirements")
    except Exception as e:
        print (e)
        print(f"Could not gather data on {stock}")

exportList = exportList.sort_values(by='RS_Rating', ascending=False)
print('\n', exportList)
writer = ExcelWriter("ImmediateOutput.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()
templist = exportList.copy()
exportList = exportList.drop(columns = exportList.columns,axis=0)
exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "5 Day MA", "5 day Low", "5 day High"])


tickers= templist['Stock']
tickers = tickers[:10]
x = pd.DataFrame(tickers)
x.to_csv('names_week.csv')

#data = yf.download(tickers=list(tickers), period='1y', interval='1d')
#data.to_csv("stock_data.csv")


###########################################################################################
## collect news companywise ##

finviz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    resp = urlopen(req)    
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')
    
        print ('\n')
        print ('Recent News Headlines for {}: '.format(ticker))
        
        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            print(a_text,'(',td_text,')')
            if i == n-1:
                break
except KeyError:
    pass

# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text() 
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]
        
        parsed_news.append([ticker, date, time, text])

news = pd.DataFrame()
# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
news['Ticker'].unique()
news= news.groupby(['Ticker']).agg({'Headline':','.join}).reset_index()
#news


# ### Sentiment Analysis

# Invoking the TFIDFVectorizer
tf_data=TfidfVectorizer()
# Copying the data into a new dataframe called vader
vader=news.copy()

sentiment=vad()
# Making additional columns for sentiment score in the vader dataframe
sen=['Positive','Negative','Neutral']
sentiments=[sentiment.polarity_scores(i) for i in vader['Headline'].values]
vader['Negative Score']=[i['neg'] for i in sentiments]
vader['Positive Score']=[i['pos'] for i in sentiments]
vader['Neutral Score']=[i['neu'] for i in sentiments]
vader['Compound Score']=[i['compound'] for i in sentiments]
score=vader['Compound Score'].values
t=[]
for i in score:
    if i >=0.05 :
        t.append('Positive')
    elif i<=-0.05 :
        t.append('Negative')
    else:
        t.append('Neutral')
vader['OverallSentiment']=t
vader.index = range(1,11,1) 

import re
stop_words = stopwords.words()
remove_words = ['AMAT','Applied','Materials',"Stock","Stocks","etsy","stocks","stock","freeportmcmoran","fcx",
               "viacomcbs","discovery","rentals","uri","united","paypal","2020","streaming","disca","earnings","inc"]
stop_words = remove_words + list(stop_words)

def cleaning(text):        
    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)     
    
    # removing the stop-words          
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)
    text = filtered_sentence
    
    return text

news['Headline'] = news['Headline'].apply(cleaning)

pd.DataFrame(vader).to_csv('vader.csv')
pd.DataFrame(news).to_csv('news.csv')
