
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import time
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as gr
from datetime import timedelta
from datetime import datetime 
import copy
yf.pdr_override()

st.sidebar.image("STRYKER LEAP.png")
colors = ['r', 'g', 'c', 'm', 'y', 'k', 'w', 'b','r','g']
################## Using Cache to retrieve data faster ##################################
     

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def foo(bar):
    data = pd.read_csv(bar)
    return data 

#########################################################################################

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.xmple.com/wallpaper/one-colour-violet-solid-color-single-plain-2560x1440-c-ddbefa-f-24.svg") }
    </style>
    """,
    unsafe_allow_html=True
)
                                      
##########################################################################################
# Parameters 
n = 10 #the # of article headlines displayed per ticker
tickers1 = copy.deepcopy(foo("names_week.csv"))
tickers1 = tickers1['Stock']
tickers = list(tickers1)

#st.balloons()

my_bar = st.progress(0)

for percent_complete in range(100):
     time.sleep(0.15)
     my_bar.progress(percent_complete + 1)
     
st.title("Your Best Weekly Stock Screener")
#st.sidebar.title("CRYPTOMANIACS - GIM")
st.sidebar.header("Top 10 Company Details")

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Choose one option',
    ('All',tickers[0],tickers[1],tickers[2],tickers[3],tickers[4],tickers[5],tickers[6],tickers[7],tickers[8],tickers[9])
)    
########################################################################################

companies_names = list(tickers)

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[0] + '.csv'
df_a = copy.deepcopy(foo(f))
df_a['Date'] = pd.to_datetime(df_a['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[1] + '.csv'
df_b = copy.deepcopy(foo(f))
df_b['Date'] = pd.to_datetime(df_b['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[2] + '.csv'
df_c = copy.deepcopy(foo(f))
df_c['Date'] = pd.to_datetime(df_c['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[3] + '.csv'
df_d = copy.deepcopy(foo(f))
df_d['Date'] = pd.to_datetime(df_d['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[4] + '.csv'
df_e = copy.deepcopy(foo(f))
df_e['Date'] = pd.to_datetime(df_e['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[5] + '.csv'
df_f = copy.deepcopy(foo(f))
df_f['Date'] = pd.to_datetime(df_f['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[6] + '.csv'
df_g = copy.deepcopy(foo(f))
df_g['Date'] = pd.to_datetime(df_g['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[7] + '.csv'
df_h = copy.deepcopy(foo(f))
df_h['Date'] = pd.to_datetime(df_h['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[8] + '.csv'
df_i = copy.deepcopy(foo(f))
df_i['Date'] = pd.to_datetime(df_i['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[9] + '.csv'
df_j = copy.deepcopy(foo(f))
df_j['Date'] = pd.to_datetime(df_j['Date'])

dfm = [df_a, df_b, df_c, df_d,df_e,df_f, df_g, df_h,df_i, df_j]
########################################################################################

# # Getting Data From FinViz
news = copy.deepcopy(foo("news.csv"))
news = news.drop('Unnamed: 0',axis = 1)

vader = copy.deepcopy(foo("vader.csv"))
vader = vader.drop('Unnamed: 0',axis = 1)

clean_data = "".join(news['Headline'].values)
word_cloud = WordCloud(width=1200,height=700,
                       background_color='beige',colormap='inferno',
                       max_words=100).generate(clean_data)

############################# NEWS Extraction ######################################################

from collections import Counter

def news_extractor(i):
     company_x= news.loc[news['Ticker'] == companies_names[i]]
     company_x= company_x['Headline']
     a = Counter(" ".join(company_x).split()).most_common(10)
     rslt_x = pd.DataFrame(a, columns=['Word', 'Frequency'])
     clean_data_x = "".join(company_x.values)
     word_cloud_x = WordCloud(width=1200,height=700,
                       background_color='beige',colormap='inferno',
                       max_words=50).generate(clean_data_x)
     return word_cloud_x

######################################################################################

i=0

dff = vader.copy()
dff['Average Closing Price Last Week'] = 0.00
for ticker in list(vader['Ticker']):   
         f = '/Users/himanshi/Desktop/project_Stock/data_week/' + ticker + '.csv'
         print(f)
         df = copy.deepcopy(foo(f))
         dff['Average Closing Price Last Week'][i] =df['Close'].mean()  
         i=i+1

v = pd.to_datetime(df['Date'])
v = v + pd.DateOffset(days=7)
v = v.dt.strftime('%Y-%m-%d') 
s = v[0]
e = v[4]

if add_selectbox=='All':
            st.subheader('Top 10 US companies stocks this week:')
            st.write("From " + s + " to " + e)
            # LSTM Part
            temp = pd.DataFrame(columns = ['Company Stock Name', 'Average Closing price Upcoming Week'])
            temp['Company Stock Name'] = companies_names
            
            for i in range(0,10,1):
                       df1 = copy.deepcopy(foo(f'/Users/himanshi/Desktop/project_Stock/data_pred/{companies_names[i]}.csv'))
                       temp['Average Closing price Upcoming Week'][i] = df1['Predicted Closing Price'].mean()
            ######################################## Graphs #########################################
            plt.figure(figsize = (15,10))            
            companies_to_show = [0,1,2,3,4,5,6,7,8,9] 
            go = dff.merge(temp,left_on='Ticker', right_on='Company Stock Name')
            go['Percent_Increase'] = ((go['Average Closing price Upcoming Week']- go['Average Closing Price Last Week'])/go['Average Closing Price Last Week'])*100
            go['Average Closing price Upcoming Week'] = go['Average Closing price Upcoming Week'].astype(float)
            go['Percent_Increase'] = go['Percent_Increase'].astype(float)
            go['Average Closing Price Last Week']=go['Average Closing Price Last Week'].astype(float)
            go['Average Closing price Upcoming Week'] = go['Average Closing price Upcoming Week'].round(decimals=4)
            go['Positive Score'] = go['Positive Score'].astype(float)
            go['Percent_Increase'] = go['Percent_Increase'].round(decimals=4)
            go['Positive Score'] = (go['Positive Score']* 100).map('{:,.1f}'.format) 
            go['Negative Score'] = (go['Negative Score']* 100).map('{:,.1f}'.format)
            go['Neutral Score'] = (go['Neutral Score']* 100).map('{:,.1f}'.format)
            go['Average Closing Price Last Week']=go['Average Closing Price Last Week'].round(decimals=4)
            #table
            fig = gr.Figure(data=[gr.Table(
                header=dict(values=list(['Ticker','Average Closing Price Last Week','Average Closing price Upcoming Week','Percent_Increase','Positive Score','Negative Score','Neutral Score']),
                            fill_color='paleturquoise',
                            align='left',
                            font=dict(size=16)),
                cells=dict(values=[go['Ticker'],go['Average Closing Price Last Week'],go['Average Closing price Upcoming Week'],go['Percent_Increase'],go['Positive Score'],go['Negative Score'],go['Neutral Score']]
                           ,fill_color='lavender',
                           align='left',
                           height=30,
                           font=dict(size=14)))
            ])
            fig.update_layout(autosize=False,width=1200, height=400,
                              margin=dict(
                                    l=10,
                                    r=10,
                                    b=10,
                                    t=10,
                                    pad=0
                                     ),
                              paper_bgcolor="LightSteelBlue",
                               )
            st.plotly_chart(fig)
            
            # bar chart
            st.markdown("***Average Closing Price Upcoming Week for Top 10 Company Stocks***")
            fig = px.bar(go, x='Ticker', y='Average Closing price Upcoming Week')
            fig.update_layout(autosize=True,#,width=1000, height=380,
                              margin=dict(
                                    l=8,
                                    r=8,
                                    b=8,
                                    t=8,
                                    pad=0
                                     ),
                              paper_bgcolor="paleturquoise",
                               )
            st.plotly_chart(fig)
            
            st.markdown("***Percentage Increase in Average Closing Price in Upcoming Week Compared to Last week for Top 10 Company Stocks***")
            fig = px.bar(go, x='Ticker', y='Percent_Increase')
            fig.update_layout(autosize=True,#,width=1000, height=380,
                              margin=dict(
                                    l=8,
                                    r=8,
                                    b=8,
                                    t=8,
                                    pad=0
                                     ),
                              paper_bgcolor="paleturquoise",
                               )
            st.plotly_chart(fig)
            # line chart
            st.markdown("***Predicted Stock Closing Price for Top 10 Company Stocks***")
            i=0
            for company in companies_names:
                    df1 = copy.deepcopy(foo(f'/Users/himanshi/Desktop/project_Stock/data_pred/{company}.csv')) 
                    df1['Date'] = v
                    plt.plot(df1['Date'],df1['Predicted Closing Price'], color=colors[i])
                    plt.plot(0,df1['Predicted Closing Price'][0] ,color=colors[i], label='predict_{}'.format(company)) #only to place the label
                    i=i+1
                    
            plt.legend(loc='best')
            plt.show()   
            st.pyplot()   
            
            st.markdown("***Some Common Words in News for Top 10 Company Stocks***")
            plt.figure(figsize=[10,15])
            plt.title("Top 100 most used words")
            plt.imshow(word_cloud)
            plt.show()
            st.pyplot()
            
            
            
dic = {companies_names[0]:0,companies_names[1]:1,companies_names[2]:2,companies_names[3]:3,
        companies_names[4]:4,companies_names[5]:5,companies_names[6]:6,companies_names[7]:7,
        companies_names[8]:8,companies_names[9]:9}           
 
def make_table(df1,i):
            st.markdown("***Predicted Closing Price Upcoming Week***")
            fig = gr.Figure(data=[gr.Table(
                          header=dict(values=['Date','Predicted Closing Price'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=16)),
            cells=dict(values=[df1['Date'],df1['Predicted Closing Price']],
            fill_color='lavender',
            align='left',
            height=30,
            font=dict(size=14)))
            ])
            fig.update_layout(autosize=False,width=400, height=230,
              margin=dict(
                    l=10,
                    r=10,
                    b=10,
                    t=10,
                    pad=0
                     ),
             paper_bgcolor="LightSteelBlue",
             )
            st.plotly_chart(fig)
            #st.line_chart(df1['Predicted Closing Price'],width =500, height =500)
 
            
def make_table2(df1,i):
            st.markdown("***Deviation Last Week***")
            fig = gr.Figure(data=[gr.Table(
                        header=dict(values=['Date','Actual Closing Price','Predicted Closing Price','Percent deviation'],
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(size=16)),
                        cells=dict(values=[df1['Date'],df1['Actual Closing Price'],df1['Predicted Closing Price'],df1['Percent deviation']],
                        fill_color='lavender',
                        align='left',
                        height=30,
                        font=dict(size=14)))
                        ])
            fig.update_layout(autosize=False,width=700, height=231,
                  margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=10,
                        pad=0
                         ),
                  paper_bgcolor="LightSteelBlue",
                  )
            st.plotly_chart(fig)
            fig1 = gr.Figure()
            fig1.add_trace(gr.Scatter(x=df1['Date'],
                    y=df1['Actual Closing Price'],
                    name = 'Actual Closing Price',
                    connectgaps=True ))
            fig1.add_trace(gr.Scatter(x=df1['Date'],
                    y=df1['Predicted Closing Price'],
                    name = 'Predicted Closing Price',
                    connectgaps=True ))
            fig1.update_layout(autosize=True,
                  margin=dict(
                        l=20,
                        r=10,
                        b=10,
                        t=10,
                        pad=0
                         ),
                  paper_bgcolor="lavender",
                  )
            st.plotly_chart(fig1)
 
            
            
def show_fun(df, i):
                x = st.multiselect('Choose type of stock price:', ['Open','Close','Adj Close'])
                st.markdown("Past One Year Performance Data Chart")
                #df['Date'] = pd.to_datetime(df['Date'])
                #df.set_index('Date', inplace=True)
                st.line_chart(df[x])
                st.markdown("***Some Common Words in News***")
                plt.figure(figsize=[8,8])
                plt.title("Top 50 most used words")
                wc = news_extractor(i)
                plt.imshow(wc)
                fig = plt.show()
                st.pyplot(fig)
 
    
 
if add_selectbox!='All':
            st.header(add_selectbox)
            st.subheader("Last Week Statistics")   
            temp1 = copy.deepcopy(foo(f'/Users/himanshi/Desktop/project_Stock/data_week/{add_selectbox}.csv'))
            temp1[['Open','High','Low','Close','Adj Close']] = temp1[['Open','High','Low','Close','Adj Close']].round(decimals=4)
            # table 1
            fig = gr.Figure(data=[gr.Table(
                       header=dict(values=['Date','Open','High','Low','Close','Adj Close','Volume'],
                                   fill_color='paleturquoise',
                                   align='left',
                                   font=dict(size=16)),
                       cells=dict(values=[temp1['Date'],temp1['Open'],temp1['High'],temp1['Low'],temp1['Close'],temp1['Adj Close'],temp1['Volume']],
                                  fill_color='lavender',
                                  align='left',
                                  height=30,
                                  font=dict(size=14)))])
     
            fig.update_layout(autosize=False,width=1000, height=200,
                      margin=dict(
                            l=10,
                            r=10,
                            b=10,
                            t=10,
                            pad=0
                             ),
                      paper_bgcolor="LightSteelBlue",
                       )
            st.plotly_chart(fig)
            df1 = copy.deepcopy(foo(f'/Users/himanshi/Desktop/project_Stock/data_pred/{add_selectbox}.csv')) 
            df1['Date'] = v
            # table 2
            make_table(df1,dic[add_selectbox]) 
            # table 3
            df0 = copy.deepcopy(foo(f'/Users/himanshi/Desktop/project_Stock/data_old_pred/{add_selectbox}.csv'))
            tp = (df0['Predicted Closing Price']- df0['Actual Closing Price'])/df0['Actual Closing Price'] * 100
            df0['Percent deviation'] = tp.map('{:,.2f}'.format)
            df0['Date'] = pd.to_datetime(temp1['Date'])
            df0['Date'] = df0['Date'].dt.strftime('%Y-%m-%d')
            make_table2(df0,i)
            show_fun(dfm[dic[add_selectbox]],dic[add_selectbox])



           




           

