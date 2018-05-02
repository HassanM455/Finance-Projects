import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


covarMat = pd.read_csv('/home/hassan/Downloads/Book1.csv')
returns =  pd.read_csv('/home/hassan/Downloads/Book2.csv')
tBills = pd.read_csv('/home/hassan/Downloads/Book3.csv')
tBills.drop(0,inplace=True)

sP500 = pd.read_csv('/home/hassan/Downloads/Book4.csv')
asset_pctChange_daily = pd.read_csv('/home/hassan/Downloads/Book5.csv')
asset_pctChange_daily.drop('S&P500',axis=1,inplace=True)
omega = np.array(covarMat)
returns = returns[['Stock code' , 'avg_expected_return']]
exp_returns = np.array(returns['avg_expected_return'])
assets = 20
assets_ref = 2
asset_pctChange = np.array(asset_pctChange_daily)




risk_free_rate = 0.021
tBills.drop('Unnamed: 2', axis=1, inplace=True)

for i in range(len(tBills)):
	tBills.loc[i+1,'Tbills'] = tBills.loc[i+1,'Tbills']/100



tBills_sP500 = tBills.merge(sP500, on='Time Period')
pctCh_sP500 = tBills_sP500['S&P 500'].pct_change()
df_pctCh_sP500 = pd.DataFrame({'pctCh_sP500': pctCh_sP500})


pctCh_sP500_tBills = df_pctCh_sP500.join(tBills_sP500['Tbills'])

ch1 = pctCh_sP500_tBills.describe()
ch2 = pctCh_sP500_tBills.corr()
corr_ref = ch2.loc['Tbills','pctCh_sP500']


means_ref = np.array(ch1.loc['mean',ch1.columns])
std_ref = np.array(ch1.loc['std',ch1.columns])
labels_ref = np.array(ch1.columns)
ref_cov = corr_ref*std_ref[0]*std_ref[1]


print('        ' , labels_ref)
print('mean  : ' , means_ref)
print('std   : ' , std_ref)
tbill_ret = means_ref[1]
sP500_ret = means_ref[0]
tbill_risk = std_ref[1]
sP500_risk = std_ref[0]

print('reference portfolio covariance : ', ref_cov)
'''________________Monte Carlo Simulation_________________'''
numOfPortfolios = 75000
returns_risks = np.array([[0,0]])
weights_hist = np.array([[0 for i in range(assets)]])
sharpeRatios = np.array([0])



returns_risks_ref = np.array([[0,0]])
weights_hist_ref = np.array([[0 for i in range(assets_ref)]])
sharpeRatios_ref = np.array([0])
covarMat_ref = np.array(pctCh_sP500_tBills.cov())


'''___ Monte Carlo Simulation for Optimal Portfolio calculation___'''
for i in range(numOfPortfolios):
	w = np.random.random(size=assets)
	w = np.absolute(w)
	w /= sum(w)
	first = omega.dot(w)
	variance = np.dot(w, first)
	volatility = np.sqrt(variance)
	return_portfolio = np.dot(w,exp_returns)
	diff = return_portfolio - risk_free_rate
	sharpeRatios = np.append(sharpeRatios, diff/volatility)
	returns_risks = np.append(returns_risks,
		                      [[return_portfolio,volatility]],
		                      axis=0)
	weights_hist = np.append(weights_hist,
						      [[i for i in w]],
						      axis=0)
else : 
	returns_risks = np.delete(returns_risks,(0),axis=0)
	weights_hist = np.delete(weights_hist,(0),axis=0)
	sharpeRatios = np.delete(sharpeRatios,(0),axis=0)
	max_sharpe = max(sharpeRatios)
	max_sharpe_index = np.argmax(sharpeRatios)
	weights_optimal = weights_hist[max_sharpe_index]


exp_returns_ref = np.array([sP500_ret , tbill_ret])

for i in range(numOfPortfolios):
	w_ref = np.random.random(size=assets_ref)
	w_ref = np.absolute(w_ref)
	w_ref /= sum(w_ref)
	first_ref = np.dot(covarMat_ref,w_ref)
	variance_ref = np.dot(w_ref, first_ref)
	volatility_ref = np.sqrt(variance_ref)
	return_portfolio_ref = np.dot(w_ref,exp_returns_ref)
	diff_ref = return_portfolio_ref - risk_free_rate
	sharpeRatios_ref = np.append(sharpeRatios_ref, diff_ref/volatility_ref)
	returns_risks_ref = np.append(returns_risks_ref,
		                      [[return_portfolio_ref, volatility_ref]],
		                      axis=0)
	weights_hist_ref = np.append(weights_hist_ref,
						      [[i for i in w_ref]],
						      axis=0)
else : 
	returns_risks_ref = np.delete(returns_risks_ref,(0),axis=0)
	weights_hist_ref = np.delete(weights_hist_ref,(0),axis=0)
	sharpeRatios_ref = np.delete(sharpeRatios_ref,(0),axis=0)
	max_sharpe_ref = max(sharpeRatios_ref)
	max_sharpe_index_ref = np.argmax(sharpeRatios_ref)
	weights_optimal_ref = weights_hist_ref[max_sharpe_index_ref]


	
print('\n \n')

print('the maximum sharpe ratio is : ', max_sharpe)
print('optimal returns & risk : ' , returns_risks[max_sharpe_index])
return_risk_opt = returns_risks[max_sharpe_index]
return_risk_refOpt = returns_risks_ref[max_sharpe_index_ref]

print('\n')
w1_ref_port = returns_risks[max_sharpe_index,0]/tbill_ret
print(str(returns_risks[max_sharpe_index,0])+'/'+str(tbill_ret))
print('\n')
print('The reference portfolio needs a '+str(w1_ref_port)+' times greater investment to match our portfolio\'s return')
print('\n')
risk_ref = w1_ref_port*tbill_risk
riskComparison_port2ref = returns_risks[max_sharpe_index,1]/risk_ref
print('Our portfolio is '+str(riskComparison_port2ref)+' times risker than the reference portfolio')
print(str(returns_risks[max_sharpe_index,1])+'/'+str(risk_ref))

print('\n \n')
print('the maximum sharpe ratio for ref portfolio is : ', max_sharpe_ref)
print('optimal returns & risk for ref portfolio is : ' , returns_risks_ref[max_sharpe_index_ref])
print('weights for optimal ref portfolio are : ' , weights_optimal_ref)
print('\n \n ')
print('comparing sharpe ratios : ' , max_sharpe , max_sharpe_ref )



portfolio_pctChange = asset_pctChange.dot(weights_optimal)
portfolio_pctChange_ref = np.array(pctCh_sP500_tBills['Tbills']*w1_ref_port)

df_pctCh_Optport_ref = pd.DataFrame({'Opt Pf Pct Change' : portfolio_pctChange ,
	                                 'Ref Pf Pct Change' : portfolio_pctChange_ref})


sharpe_ratio_ref_m = (return_risk_opt[0]-risk_free_rate)/risk_ref


df_corr_portRef = df_pctCh_Optport_ref.corr()
df_covMat_portRef = df_pctCh_Optport_ref.cov()
correlation_OptPort_ref = df_corr_portRef.loc['Opt Pf Pct Change' , 'Ref Pf Pct Change']


df_weights_optimal = pd.DataFrame({'optimal port weights' : weights_optimal})
df_weights_optimal_ref = pd.DataFrame({'optimal ref_port weights' : weights_optimal_ref})

df_return_risk_opt = pd.DataFrame({'optimal portfolio'   : np.append(return_risk_opt,max_sharpe),
	                           'ref optimal portfolio' : np.append(return_risk_refOpt,max_sharpe_ref),
	                           'ref retMatch portfolio' : [return_risk_opt[0], risk_ref, sharpe_ratio_ref_m ]},
	                            index = ['return','risk','sharpe ratio'] )

df_misc = pd.DataFrame({ 'Others' : [correlation_OptPort_ref , w1_ref_port , riskComparison_port2ref] },
	                   index = ['corr OptPort & Ref_m' , 'return scale OptPort/refPort', 'risk scale OptPort/refPort'])

df_corr_portRef.to_csv('corr_matrix_optPort_ref_m.csv')
df_covMat_portRef.to_csv('VaCov_mat_optPort_ref_m.csv')
df_weights_optimal.to_csv('optPort_weights.csv')
df_weights_optimal_ref.to_csv('optRef_weights.csv')
df_misc.to_csv('misc.csv')
df_return_risk_opt.to_csv('ret_risk_sharpe_optPort_optref_mref.csv')


print(df_return_risk_opt)

x =  np.linspace(0, 0.4, numOfPortfolios)
x2 =  np.linspace(0, 0.05, numOfPortfolios)

plt.plot(x, max_sharpe*x + risk_free_rate,label='Our Capital Market Line' ,color='black')
plt.plot(x2, max_sharpe_ref*x2 + risk_free_rate,label='ref Capital Market Line' ,color='green')



plt.scatter(return_risk_opt[1],
			return_risk_opt[0],
			label='Optimal Portfolio',
			c='c',
			marker='x',
			s=200)
plt.scatter(returns_risks[:,1],
	        returns_risks[:,0],
	        label='BitCoin Portfolio Frontier',
	         c='r',
	         marker='o')
plt.scatter(returns_risks_ref[:,1],
	        returns_risks_ref[:,0],
	        label='Reference Portfolio Frontier',
	         c='b',
	         marker='o')


plt.xlabel('Risk')
plt.ylabel('Expected Return')
plt.legend(loc=4, prop={'size': 12})
plt.title('Efficient Frontiers')
print(return_risk_opt)
plt.show()



