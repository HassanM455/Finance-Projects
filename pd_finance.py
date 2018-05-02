import pandas as pd
from pandas_datareader import data
import datetime
import numpy as np
import matplotlib.pyplot as plt 

start_date = input('Enter start date in year-month-date format : ')
end_date = input('Enter end date in year-month-date format : ')


stock = input('Enter stock code : ' )
print('Select source from the following options \n')
print('_________display choices__________ i.e. iex \n')
source = input() 

f = data.DataReader(stock, source, start_date, end_date)
date_time = np.array(f.index)
print(f.index)

values = np.array(f.close)


val1 = input('Would you like to see the daily percent change? Enter Yes or No : ')
flag = True
val1 = val1.lower()

while flag:
	if val1 == 'yes':
		print(f.pct_change().close)
		flag = False
	elif val1 == 'no': 
		flag = False
	else:
		val1 = input('Please enter Yes or No : ') 
		
print('\n')

val2 = input('Would you like to plot the stock history? Enter Yes or No : ')
flag2 = True
val2 = val2.lower()

while flag2:
	if val2 == 'yes':
		plt.plot(date_time,values)
		plt.title('Stock history of ' + stock)
		plt.xlabel('Date')
		plt.ylabel('Adjusted closing price')
		plt.show()
		flag2 = False
	elif val2 == 'no': 
		flag2 = False
	else:
		val2 = input('Please enter Yes or No : ') 

