import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

my_df = pd.read_csv('SPY, GLD, TLT Data for Risk Parity.csv')

# making the date the index
my_df['Date'] = pd.to_datetime(my_df['Date'], dayfirst=True)
my_df = my_df.set_index('Date')
#port_daily_return = port_daily_return.set_index('Date')

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

# plots the allocation to each asset class over time


       
#print(contribution)

# Showing growth of hypothetical $1,000 investment into the strategy
portfolio_df['Portfolio Value'] = ((portfolio_df['Portfolio Return'] + 1).cumprod())*100

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

# plots the gross exposure of the portfolio over time
(portfolio_df['TLT Alloc'] + portfolio_df['GLD Alloc'] + portfolio_df['SPY Alloc']).plot()
plt.title('Gross Portfolio Exposure')
plt.show()

# calculates monthly returns for the strategy
monthly = portfolio_df['Portfolio Value'].resample('BM').apply(lambda x: x[-1])
new1 = monthly['2015':'2019'].pct_change()
fig, ax = plt.subplots()
new1.plot(kind='bar', ax=ax)
ax.bar(new1.index, new1)
ax.autoscale_view()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Strategy Monthly Performance')
plt.show()

# Calculating calendar year returns for the strategy
yearly = portfolio_df['Portfolio Value'].resample('Y').apply(lambda x: x[-1])
new2 = yearly.pct_change()
fig, ax = plt.subplots()
new2.plot(kind='bar', ax=ax)
ax.bar(new2.index, new2)
ax.autoscale_view()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Strategy Annual Performance')
plt.show()

#Return contribution graph for daily, monthly, annual and since inception values
TLT = []
Portfolio = []
GLD = []
SPY = []
Year = list(range(2004, 2021))
contribution = {'Year': Year}
contribution = pd.DataFrame(contribution)
y = 2006
t=g=s=p=0
for index, row in portfolio_df.iterrows():
    if index.year == y:
        t += row['TLT Return']
        g += row['GLD Return']
        s += row['SPY Return']
        p += row['Portfolio Return']
    else:
        TLT.append(t)
        Portfolio.append(p)
        GLD.append(g)
        SPY.append(s)
        t=p=g=s=0
        y=index.year
        t += row['TLT Return']
        g += row['GLD Return']
        s += row['SPY Return']
        p += row['Portfolio Return']
TLT.append(t)
Portfolio.append(p)
GLD.append(g)
SPY.append(s)
contribution.insert(1,'TLT', TLT)
contribution.insert(2,'GLD', GLD)
contribution.insert(3,'SPY', SPY)
contribution.insert(4,'Portfolio', Portfolio)
contribution = contribution.set_index('Year')

contribution.plot.bar()
plt.title('Contribution of TLT, GLD, SPY')
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
risk_free['Date'] = pd.to_datetime(risk_free['Date'], dayfirst=True)

daily_interest_rate = []
for index, row in my_df.iterrows():
    date = datetime.datetime(index.year, index.month, 1)
    for index, row in risk_free.iterrows():
        if date == row['Date']:
            daily_interest_rate.append(row['3 Month Treasury Rate']/3000)
my_df.insert(6, 'Daily Interest Rate', daily_interest_rate)

my_df['Rolling Sharp Ratio'] = (portfolio_df['Portfolio Return'] - my_df['Daily Interest Rate'])/(portfolio_df['Portfolio Return'].rolling(252).std())
#print(my_df['Rolling Sharp Ratio'])
my_df['Rolling Sharp Ratio'].plot()
plt.axhline(y=my_df['Rolling Sharp Ratio'].mean(), color="red")

plt.title('Rolling Sharpe Ratio')
plt.show()

#plots return, drawdown and standard deviation of the strategy in type of percent
ax = (portfolio_df['Drawdown']*100).plot()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Drowdown Percentage')
plt.show()

ax = (portfolio_df['Portfolio Return']*100).plot()
plt.axhline(y=(portfolio_df['Portfolio Return']*100).mean(), color="red")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Portfolio Return Percentage')
plt.show()

ax = (port_stdev*100).plot()
plt.axhline(y=(port_stdev*100).mean(), color="red")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Standard Deviation Percentage')
plt.show()






