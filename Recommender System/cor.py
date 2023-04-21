
import numpy as np

def rmse(predictions,targets):
	rms = 0
	count = 0
	for i in range(len(predictions)):
		if targets[i]!=0 :
			rms = rms + (predictions[i] - targets[i])**2
			count = count + 1
			
	return np.sqrt(rms/count)
	

def rank(x):
	rank_X = []
	for i in range(len(x)):
		r = 1
		s = 1
		for j in range(i):
			if x[j] < x[i]:
				r = r+1
			if x[j] == x[i]:
				s = s+1
		for j in range(i+1,len(x)):
			if x[j] < x[i]:
				r = r+1
			if x[i] == x[j]:
				s = s+1
		rank_X.append(r+(s-1)*0.5)
	return rank_X
	
def spearman(x,y):
	
	# print(rank_X)
	# print(rank_Y)
	sum_x = 0
	sum_y = 0
	sum_xy = 0
	sqsum_x = 0
	sqsum_y = 0
	
	for i in range(len(x)):
		sum_x = sum_x + x[i]
		sum_y = sum_y + y[i]
		sum_xy = sum_xy + x[i]*y[i]
		sqsum_x = sqsum_x + x[i]*x[i]
		sqsum_y = sqsum_y + y[i]*y[i]
	
	cor = (len(x)*sum_xy - sum_x*sum_y)/((len(x)*sqsum_x - sum_x*sum_x)*(len(x)*sqsum_y - sum_y*sum_y))**0.5
	return cor
		
	
# x = [15,18,21,15,21]
# y = [25,25,27,27,27]
# rank_X = rank(x)
# rank_Y = rank(y)
# sp = spearman(rank_X,rank_Y)
# print(sp)



