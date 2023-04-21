import numpy as np
from numpy import linalg as lg
import xlrd
import sklearn.model_selection as ms
import math
import time
from cor import rmse,spearman,rank
import random

no_of_users = 6045
no_of_movies = 3955

RMSE = 0
PrecisionOTK=0
SpearmanRC=0
TimeTaken=0

loc=("ratings.xlsx")				#Reading the Excel File

ratings=np.zeros(shape = (6045,3955))
# ratings=np.zeros(shape = (3,3))
wb=xlrd.open_workbook(loc) 
sheet=wb.sheet_by_index(0) 
sheet.cell_value(0, 0)
# print(sheet.nrows)

for i in range(sheet.nrows):
	user=(int)(sheet.cell_value(i,0))
	movie=(int)(sheet.cell_value(i,1))
	star=sheet.cell_value(i,2)
	ratings[user - 1][movie - 1]=star		#Storing the Ratings such that users are represented by rows and movies across columns

# print("ratings:",ratings)


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
	# print(basis)
    return np.array(basis)



def reduced_form(U,sigma,V):
	total_square_sum=0
	for i in range(no_of_movies):
		total_square_sum = total_square_sum + sigma[i,i]**2

	temp_square_sum=0

	for i in range(no_of_movies):
		temp_square_sum = temp_square_sum + sigma[i,i]**2

		if(temp_square_sum/total_square_sum >= 0.9):			#Break if energy is less than 90%
			break
	
	# print("U:",U.shape)
	# print("Sigma:",sigma.shape)
	# print("V:",V.shape)

	newU = U[:,:i+1]
	newSigma = sigma[:i+1,:i+1]
	newV = V[:i+1,]

	return newU,newSigma,newV


def SVD(ratings):

	AAT = np.matmul(ratings,ratings.transpose()) # A * A(transpose)
	#print(AAT)
	
	Ueigenvalues, U= lg.eig(AAT)
	U = np.real(U)
	
	Ueigenvalues = np.real(Ueigenvalues)

	# print("Ueigenvalues:",Ueigenvalues)
	sorted_Ueigenvalues_indices = np.flip(np.argsort(Ueigenvalues))			
	# print(sorted_Ueigenvalues_indices)
	# print("sorted_Ueigenvalues_indices:",sorted_Ueigenvalues_indices)
	U=U[sorted_Ueigenvalues_indices]
	
	ATA = np.matmul(ratings.transpose(),ratings) #A(transpose) * A
	Veigenvalues, V= lg.eig(ATA)
	V = np.real(V)
	# print(V)
	Veigenvalues = np.real(Veigenvalues)
	sorted_Veigenvalues_indices = np.flip(np.argsort(Veigenvalues))
	
	V = V[sorted_Veigenvalues_indices]
	Veigenvalues = Veigenvalues[sorted_Veigenvalues_indices]

	# V = V.transpose()

	sigma_matrix = np.zeros(shape = ratings.shape)

	if(Ueigenvalues.shape[0]<Veigenvalues.shape[0]):				#Calculating Sigma
		lim=Ueigenvalues.shape[0]
	else:
		lim=Veigenvalues.shape[0]
	for i in range (0,lim,1):
		if(Ueigenvalues[i]>=0):
			sigma_matrix[i][i]=math.sqrt(Ueigenvalues[i])	
	
	return U,sigma_matrix,V

def main():
	start= time.time()
	U,sigma,V=SVD(ratings)
	nU,nsigma,nV= reduced_form(U,sigma,V)

	predicted_ratings = np.zeros((no_of_users,no_of_movies)) #These will be predicted ratings for training set using U,sigma,VT
	predicted_ratings = np.matmul(U, np.matmul(sigma,V))
	new_predicted_ratings = np.zeros((no_of_users,no_of_movies)) #These will be predicted ratings for training set using U,sigma,VT
	new_predicted_ratings = np.matmul(nU, np.matmul(nsigma,nV))
	
	
	pr_list=[]
	for i in np.array(ratings).flat :
		pr_list.append(i)
	npr_list=[]
	for i in np.array(predicted_ratings).flat :
		npr_list.append(i)
	nnpr_list=[]
	for i in np.array(new_predicted_ratings).flat :
		nnpr_list.append(i)
	rank_ratings = pr_list
	rank_predicted = npr_list
	rank_new_predicted = nnpr_list
	cor1 = spearman(rank_ratings,rank_predicted)			#Calculating Correlation
	cor2 = spearman(rank_ratings,rank_new_predicted)

	print(cor1 )
	print(cor2 )
	rootmeansquareerror = rmse(pr_list,npr_list)			#Calculating rmse
	print(rootmeansquareerror )
	newrootmeansquareerror = rmse(pr_list,nnpr_list)
	print(newrootmeansquareerror )
	
	precision=0
	precision2=0
	
	for j in range(100):							#Calculating Precision
		x=random.randint(0,no_of_users-1)
		a=ratings[x]
		b=predicted_ratings[x]
		c=new_predicted_ratings[x]
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

	end = time.time()



	print(end - start)
	
main()