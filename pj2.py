import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go






stocks_df=pd.read_csv('stocks_dataset.csv')
stocks_df                                     #reading the stock data file
                 
                 
                 
stocks_df.info()                              #getting dataframe info


# Function to normalize the prices based on the initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i]=x[i]/x[i][0]
        
        
    return x
    
    
    

# Function to plot interactive plot
def intr_plot(df,title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'],y =df[i],name =i)
    
    
    fig.show()
    
# Function to calculate the daily returns 

def daily_return(df):
    df_daily_return=df.copy()
    
    for i in df.columns[1:]:
        for j in range(1,len(df)):
            df_daily_return[i][j]=(df[i][j]-df[i][j-1])/(df[i][j-1])
            
        df_daily_return[i][0]=0
    return df_daily_return
    
    
    
beta, alpha = np.polyfit(s['sp500'], s['TSLA'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('TSLA', beta, alpha))  



#  plot the scatter plot and the straight line on one plot
s.plot(kind = 'scatter', x = 'sp500', y = 'TSLA', color = 'w')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(s['sp500'], beta * s['sp500'] + alpha, '-', color = 'r')


rm=s['sp500'].mean()*252




# We Assume risk free rate is zero
# Also one can use the yield of a 10-years U.S. Government bond as a risk free rate
rf=0

    
ER_tsla=rf+(beta*(rm-rf)) 

beta = {}
alpha = {}

# Loop on every stock daily return
for i in s.columns:

  # Ignoring the date and S&P500 Columns 
  if i != 'Date' and i != 'sp500':
    # plot a scatter plot between each individual stock and the S&P500 (Market)
    s.plot(kind = 'scatter', x = 'sp500', y = i, color = 'w')
    
    # Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
    b, a = np.polyfit(s['sp500'], s[i], 1)
    
    plt.plot(s['sp500'], b * s['sp500'] + a, '-', color = 'r')
    
    beta[i] = b
    
    alpha[i] = a
    
    plt.show()



