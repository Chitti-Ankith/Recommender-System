import numpy as np
from numpy import linalg as lg
import xlrd
import random
from random import choices
import time
import cor
# from svd import SVD

start_time = time.time()

def sq_sum(S):			
	sum = 0
	for i in range(0,min(S.shape)):
		sum = sum + S[i][i]**2
	
	return sum


def red(U,S,V):			#To reduce the matrix
	
	limit = min(S.shape)
	tsum = sq_sum(S)
	req = 1
	for i in range(limit,0,-1):
		current_sum = sq_sum(S[range(i)])
		if current_sum/tsum < 0.9:			#Break if 'energy' becomes less than 90%
			req = i + 1
			break
	
	V=V.transpose()
	Unew=U[:,range(req)]
	Sn=np.diag(S[range(req),range(req)])
	Vnew=V[range(req),:]
	Vnew=Vnew.transpose()
	return Unew,Sn,Vnew
		
			
def CUR(mat,n,flag = False):
	csum = np.sum(mat**2,axis = 0)
	rsum = np.sum(mat**2,axis = 1)
	# print(csum)
	tsum = np.sum(mat**2)
	cw = csum/tsum			#Probability Dist of Columns
	rw = rsum/tsum
	# print(rw)
	optionsC = [i for i in range(3952)]
	optionsR = [i for i in range(6040)]
	# print(optionsC)
	columns = choices(optionsC,cw,k=n)			#Picks the columns and rows based on their prob distribution
	rows = choices(optionsR,rw,k=n)

	R = mat[rows]
	C = mat[:,columns]
	Rf = R
	Cf = C
	# print(R)
	it = 0											#Dividing the selected rows and columns by the square root of their prob dist and number of rows and columns resp
	for i in rows:
		for j in range(3952):
			# print(R[it][j],i,j,rw[i])					
			Rf[it][j] = R[it][j]/(n*rw[i])**0.5
		it = it + 1
	
	it = 0	
	# print(Rf)
	for i in columns:
		for j in range(6040):
			Cf[j][it] = C[j][it]/(n*cw[i])**0.5
		it = it + 1
	
	
	W = []
	for i in rows:
		emptyRow=[]
		for j in columns:
			# print(mat[int(i)][int(j)])
			emptyRow.append(mat[int(i)][int(j)])
		W.append(emptyRow)
	W=np.array(W)

	X,s_temp,Y=np.linalg.svd(W)
	S=np.zeros(shape=W.shape)
	Y = np.transpose(Y)
	
	for i in range(0,s_temp.shape[0],1):
		S[i][i]=s_temp[i]
		
	if flag == True:          #If you want to reduce the CUR matrix
		X,S,Y = red(X,S,Y)
		n = min(S.shape)

	
	for i in range(n):
		if S[i][i] != 0 and S[i][i] > 0.1:
			S[i][i] = 1/(S[i][i])
		else:
			S[i][i] = 0
			
	
	U = np.matmul(np.matmul(Y,S),np.transpose(X))
	# U = np.matmul(np.matmul(Y,(np.matmul(S,S))),np.transpose(X))
	return (Cf,U,Rf)
	
	
	
no_of_users = 6040
no_of_movies = 3952
loc=("ratings.xlsx")			#Reading the Excel File

ratings=np.zeros(shape = (6040,3952))
wb=xlrd.open_workbook(loc) 
sheet=wb.sheet_by_index(0) 
sheet.cell_value(0, 0)


for i in range(sheet.nrows):
	user=(int)(sheet.cell_value(i,0))
	movie=(int)(sheet.cell_value(i,1))
	star=sheet.cell_value(i,2)
	ratings[user - 1][movie - 1]=star		#Storing the Ratings such that users are represented by rows and movies across columns

print("ratings:",ratings)

rat_list = []
for i in np.array(ratings).flat:
	rat_list.append(i)


	
C,U,R = CUR(ratings,1000) #Calling the CUR Function
Cr,Ur,Rr = CUR(ratings,1000,True)	#To reduce the CUR matrix


#%%%%%%%%% precision %%%%%%%%%%%%%%%%%%%%%%%%


final = np.matmul(C,np.matmul(U,R))  #Approximated Matrix
final2 = np.matmul(Cr,np.matmul(Ur,Rr))
# print(final)
end_time =time.time()
time_taken = end_time - start_time   #Time taken to calculate
print("Time:",time_taken)

fin_list = []
for i in np.array(final).flat:
	fin_list.append(i)
new_fin_list=[]
for i in np.array(final2).flat:
	new_fin_list.append(i)
rms = cor.rmse(rat_list,fin_list)			#Calculating rmse and spearman correlation
sp = cor.spearman(rat_list,fin_list)
rms2= cor.rmse(rat_list,new_fin_list)
sp2 = cor.spearman(rat_list,new_fin_list)
print("rmse:",rms)
print("new rmse:",rms2)
print("Correlation:",sp)
print("new Correlation:",sp2)

# finalr = np.matmul(Cr,np.matmul(Ur,Rr))	#CUR with 90% energy
# print(finalr)
precision=0									#Calculating Precision
precision2 = 0
for j in range(100):
	x=random.randint(0,no_of_users-1)			
	a=ratings[x]
	b=final[x]
	c= final2[x]
	a=a.tolist()
	b=b.tolist()
	c=c.tolist()
	x=a
	y=b
	z=c

	pcount=0				#Precise count
	icount=0				#Rating Count
	for i in range(len(x)):
		if x[i]>3:							#Checking only for high ratings
			icount=icount+1
			if abs(y[i]-x[i]) < 1:			#Checking if diff between predicted and original rating is less.
				pcount = pcount + 1			#If yes, value is considered precise		

	
	if icount!=0:
		precision = precision + pcount/icount

	pcount=0
	icount=0
	for i in range(len(x)):	
		if x[i]>3:							#Checking only for high ratings
			icount=icount+1
			if abs(z[i]-x[i]) < 1:			#Checking if diff between predicted and original rating is less.
				pcount = pcount + 1			#If yes, value is considered precise
	if icount!=0:
		precision2 = precision2+ pcount/icount
print("precesion 1 is", precision/100 )
print("precision 2 is ",precision2/100 )
 
# end_time = time.time()
