import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

my_df = pd.read_csv('SPY, GLD, TLT Data for Risk Parity.csv')

# making the date the index
my_df['Date'] = pd.to_datetime(my_df['Date'], dayfirst=True)

my_df = my_df.set_index('Date')

# setting all prices to percentages
my_df = my_df.pct_change()

# removes first row where you can't calculate percentage change
my_df.dropna(inplace=True)

# Calculating rolling 1 year historical volatility for each asset class
my_df['TLT Vol'] = my_df['TLT'].rolling(252).std() * sqrt(252)
my_df['GLD Vol'] = my_df['GLD'].rolling(252).std() * sqrt(252)
my_df['SPY Vol'] = my_df['SPY'].rolling(252).std() * sqrt(252)

# remove calculation time for historical volatility
my_df.dropna(inplace=True)

# target volatility of the portfolio is 10% divided by 3 because there are 3 asset classes
target_vol = 0.10/3

# creates a DataFrame that calculates the allocation for various asset classes
portfolio_df = pd.DataFrame({'TLT Alloc': target_vol/my_df['TLT Vol'],
                             'GLD Alloc': target_vol/my_df['GLD Vol'],
                             'SPY Alloc': target_vol/my_df['SPY Vol'], }, index=my_df.index)

# print(portfolio_df.head())
# print(portfolio_df.tail())

# calculating the return on the asset by multiplying the assets total return by it's % weight in portfolio
portfolio_df['TLT Return'] = my_df['TLT']*portfolio_df['TLT Alloc']
portfolio_df['GLD Return'] = my_df['GLD']*portfolio_df['GLD Alloc']
portfolio_df['SPY Return'] = my_df['SPY']*portfolio_df['SPY Alloc']

# adding up the returns from SPY, TLT and GLD to create the portfolio's daily return
portfolio_df['Portfolio Return'] = portfolio_df['TLT Return'] + portfolio_df['GLD Return'] + portfolio_df['SPY Return']

# Showing growth of hypothetical $1,000 investment into the strategy
portfolio_df['Portfolio Value'] = ((portfolio_df['Portfolio Return'] + 1).cumprod())*1000

# creates a drawdown column for the strategy which we can plot
portfolio_df['Drawdown'] = portfolio_df['Portfolio Value'].div(portfolio_df['Portfolio Value'].cummax()) - 1

# plot performance of strategy
portfolio_df['Portfolio Value'].plot()
plt.title('Performance of $1,000 Investment')
plt.show()

# plot drawdowns of the strategy
portfolio_df['Drawdown'].plot()
plt.title('Strategy Drawdowns')
plt.show()

# plots the allocation to each asset class over time
portfolio_df[['TLT Alloc', 'GLD Alloc', 'SPY Alloc']].plot()
plt.title('Strategy Asset Allocation')
plt.show()

# plots the gross exposure of the portfolio over time
(portfolio_df['TLT Alloc'] + portfolio_df['GLD Alloc'] + portfolio_df['SPY Alloc']).plot()
plt.title('Gross Portfolio Exposure')
plt.show()

# calculates monthly returns for the strategy
monthly = portfolio_df['Portfolio Value'].resample('BM').apply(lambda x: x[-1])
monthly['2015':'2019'].pct_change().plot.bar()
plt.title('Strategy Monthly Performance')
plt.show()

# Calculating calendar year returns for the strategy
yearly = portfolio_df['Portfolio Value'].resample('BY').apply(lambda x: x[-1])
yearly.pct_change().plot.bar()
plt.title('Strategy Annual Performance')
plt.show()

# calculating the rolling 12 month volatility of the portfolio
port_stdev = portfolio_df['Portfolio Return'].rolling(252).std() * sqrt(252)
port_stdev.plot()
plt.title('Strategy 1 Yr Rolling Stdev')
plt.show()

# calculating the rolling 12 month return of the portfolio
rolling_return = (1 + portfolio_df['Portfolio Return']).rolling(window=252).apply(np.prod, raw=True) - 1
rolling_return.plot()
plt.title('Strategy 1 Yr Rolling Returns')
plt.show()

# calculating the rolling 12 month Sharpe Ratio of the portfolio (NOT DONE)
risk_free = pd.read_csv('Risk Free Rate.csv')

