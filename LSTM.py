import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers1 = pd.read_csv("names_week.csv")
tickers1 = tickers1['Stock']
tickers = list(tickers1)
companies_names = list(tickers)
n=10

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[0] + '.csv'
df_A = pd.read_csv(f)
df_A['Date'] = pd.to_datetime(df_A['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[1] + '.csv'
df_B = pd.read_csv(f)
df_B['Date'] = pd.to_datetime(df_B['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[2] + '.csv'
df_C = pd.read_csv(f)
df_C['Date'] = pd.to_datetime(df_C['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[3] + '.csv'
df_D = pd.read_csv(f)
df_D['Date'] = pd.to_datetime(df_D['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[4] + '.csv'
df_E = pd.read_csv(f)
df_E['Date'] = pd.to_datetime(df_E['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[5] + '.csv'
df_F = pd.read_csv(f)
df_F['Date'] = pd.to_datetime(df_F['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[6] + '.csv'
df_G = pd.read_csv(f)
df_G['Date'] = pd.to_datetime(df_G['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[7] + '.csv'
df_H = pd.read_csv(f)
df_H['Date'] = pd.to_datetime(df_H['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[8] + '.csv'
df_I = pd.read_csv(f)
df_I['Date'] = pd.to_datetime(df_I['Date'])

f = '/Users/himanshi/Desktop/project_Stock/data_log/' + companies_names[9] + '.csv'
df_J = pd.read_csv(f)
df_J['Date'] = pd.to_datetime(df_J['Date'])


num_companies = 10
look_back = 15
forward_days = 5
num_periods = 0

companies = [df_A, df_B, df_C, df_D, df_E, df_F, df_G, df_H, df_I, df_J]

for comapany in companies:
    comapany.set_index('Date', inplace=True)
    comapany.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    
### taking close price 

from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
args = []
args = [company.values.reshape(company.shape[0],1) for company in companies]
array = scl.fit_transform(np.concatenate((args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9]), axis=1))


#Get the data and splits in input X and output Y, by spliting in `n` past days as input X 
#and `m` coming days as Y.
def processData(data, look_back, foward_days,num_companies,jump=1):
        X,Y = [],[]
        for i in range(0,len(data) -look_back -foward_days +1 + 5, jump):
            X.append(data[i:(i+look_back)])
            Y.append(data[(i+look_back):(i+look_back+forward_days)])
        return np.array(X),np.array(Y)


X_test, y_test = processData(array,look_back,forward_days,num_companies)
y_test = np.array([list(a.ravel()) for a in y_test])

from keras.models import load_model
# load trained model

model = load_model('model5.h5')
Xt = model.predict(X_test)


def do_inverse_transform(output_result,num_companies = 10):
        #From input/output nootbook: apply makeup, use scl.inverse_transform and remove makeup
        
        #transform to input shape
        original_matrix_format = []
        for result in output_result:
            #do inverse transform
            original_matrix_format.append(scl.inverse_transform([result[x:x+num_companies] for x in range(0, len(result), num_companies)]))
        original_matrix_format = np.array(original_matrix_format)
        
        #restore to original shape
        for i in range(len(original_matrix_format)):
            output_result[i] = original_matrix_format[i].ravel()
    
        return output_result

def prediction_by_step_by_company(raw_model_output, num_companies=10):
        matrix_prediction = []
        for i in range(0,num_companies):
              matrix_prediction.append([[lista[j] for j in range(i,len(lista),num_companies)] for lista in raw_model_output])
        return np.array(matrix_prediction)

Xt = do_inverse_transform(Xt)

MP = prediction_by_step_by_company(Xt, n)

################################

df0= pd.DataFrame()
for i in range(10):
            df1= pd.DataFrame(columns = ['Day','Predicted Closing Price'])
            df0['Predicted Closing Price']= MP[i][MP.shape[1]-2]
            df0['Predicted Closing Price'] =df0['Predicted Closing Price'].map('{:,.2f}'.format)
            temp1 = pd.read_csv(f'/Users/himanshi/Desktop/project_Stock/data_week/{companies_names[i]}.csv')
            df0['Actual Closing Price'] = temp1[['Close']].round(decimals=3)
            df0.to_csv(f'/Users/himanshi/Desktop/project_Stock/data_old_pred/{companies_names[i]}.csv')
            df1['Predicted Closing Price'] = MP[i][MP.shape[1]-1]
            df1['Predicted Closing Price']=df1['Predicted Closing Price'].map('{:,.2f}'.format)
            df1.to_csv(f'/Users/himanshi/Desktop/project_Stock/data_pred/{companies_names[i]}.csv')