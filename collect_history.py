#### last 1 year data collection to be run once only to collect historical data
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas_datareader import data as pdr
from bs4 import BeautifulSoup
from yahoo_fin import stock_info as si
import datetime
import time
import yfinance as yf
import plotly.express as px
yf.pdr_override()
# get list of stocks

#dow_list = si.tickers_dow()
#nasdaq_list = si.tickers_nasdaq()
sp500_list = si.tickers_sp500()
sp500_list = [item.replace(".", "-") for item in sp500_list]

tickers = sp500_list # + nasdaq_list + dow_list 
#tickers = [item.replace(".", "-") for item in tickers] # Yahoo Finance uses dashes instead of dots

index_name = list(['^GSPC']) # S&P 500, Dow Jones, Nasdaq , '^IXIC',,'^DJI'
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.date.today() #- datetime.timedelta(days=1)

exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])
returns_multiples = []

################### Fetch data for 1 year indexwise ################################## 

# Calculating Index Returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_df
index_df[['Percent Change_GSPC']] = index_df['Adj Close'].pct_change()
index_df 
index_return_GSPC = (index_df['Percent Change_GSPC'] + 1).cumprod()[-1]
index_return_GSPC

################### Fetch data for 1 year companywise ################################

m = {}
pc = {}
sr = {}

def calc_return(tickers,index_return,index_name):
      for ticker in tickers:
              # Download historical data as CSV for each stock (makes the process faster)
              df = pdr.get_data_yahoo(ticker, start_date, end_date)
              x = '/Users/himanshi/Desktop/project_Stock/data_log/' + ticker + '.csv'
              df.to_csv(x)

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
calc_return(sp500_list, index_return_GSPC, "S&P500")

########################################################################################
"""# writing the multiples in csv files
import csv
with open('return_multiple.csv', 'w') as f:
    for key in m.keys():
        f.write("%s,%s\n"%(key,m[key]))

with open('percent_change.csv', 'w') as f:
    for key in pc.keys():
        f.write("%s,%s\n"%(key,pc[key]))

with open('stock_return.csv', 'w') as f:
    for key in sr.keys():
        f.write("%s,%s\n"%(key,sr[key]))
"""
########################################################################################

# Creating dataframe of only top 20% for the last week
rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.80)]
rs_stocks = rs_df['Ticker']

#################### Check if stocks performing well for last 1 year ##################

# Checking Minervini conditions of top 20% of stocks in given list
for stock in rs_stocks:    
    try:
        df = pd.read_csv(f'/Users/himanshi/Desktop/project_Stock/data_log/{stock}.csv', index_col=0)
        sma = [50, 150, 200]  # simple moving average
        for x in sma:
            df["SMA_"+str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)  # calculating sma 
        
        # Storing required values 
        currentClose = df["Adj Close"][-1]
        moving_average_50 = df["SMA_50"][-1]
        moving_average_150 = df["SMA_150"][-1]
        moving_average_200 = df["SMA_200"][-1]
        low_of_52week = round(min(df["Low"][-260:]), 2)  # 5 days a week
        high_of_52week = round(max(df["High"][-260:]), 2)
        RS_Rating = round(rs_df[rs_df['Ticker']==stock].RS_Rating.tolist()[0])
        
        try:
            moving_average_200_20 = df["SMA_200"][-20]
        except Exception:
            moving_average_200_20 = 0

        # Condition 1: Current Price > 150 SMA and > 200 SMA
        condition_1 = currentClose > moving_average_150 > moving_average_200
        
        # Condition 2: 150 SMA and > 200 SMA
        condition_2 = moving_average_150 > moving_average_200

        # Condition 3: 200 SMA trending up for at least 1 month
        condition_3 = moving_average_200 > moving_average_200_20
        
        # Condition 4: 50 SMA > 150 SMA and 50 SMA> 200 SMA
        condition_4 = moving_average_50 > moving_average_150 > moving_average_200
           
        # Condition 5: Current Price > 50 SMA
        condition_5 = currentClose > moving_average_50
           
        # Condition 6: Current Price is at least 30% above 52 week low
        condition_6 = currentClose >= (1.3*low_of_52week)
           
        # Condition 7: Current Price is within 25% of 52 week high
        condition_7 = currentClose >= (.75*high_of_52week)
        
        # If all conditions above are true, add stock to exportList
        if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7):
            exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating ,"50 Day MA": moving_average_50, "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200, "52 Week Low": low_of_52week, "52 week High": high_of_52week}, ignore_index=True)
            print (stock + " made the Minervini requirements")
    except Exception as e:
        print (e)
        print(f"Could not gather data on {stock}")

#######################################################################################

exportList = exportList.sort_values(by='RS_Rating', ascending=False)
print('\n', exportList)
writer = ExcelWriter("HistoricalData.xlsx") # past 1 year
exportList.to_excel(writer, "Sheet1")
writer.save()

#######################################################################################
# storing top 50 company names in csv which had best performance for past 1 year 

tickers= exportList['Stock']
tickers = tickers[:50]
x = pd.DataFrame(tickers)
x.to_csv('names_1yr.csv')

#######################################################################################

# This will determine top 50 names if we see long term of 1 yr 


