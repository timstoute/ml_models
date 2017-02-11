import sys
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

mydates=[]
myprices=[]

svr_lin = SVR(kernel= 'linear', C= 1e3)
svr_poly = SVR(kernel='poly', cache_size=1000, C=1e3, degree=3)
svr_rbf = SVR(kernel='rbf', cache_size=1000, C=1e3, gamma='auto')

def get_data(filename,n):
    global mydates
    global myprices
    i=0
    with open(filename, 'r') as csvfile:
        CSVFILEREADER = csv.reader(csvfile)
        next(CSVFILEREADER)
        for row in CSVFILEREADER:
            #dates.append(int(row[0].split('-')[0]))
			#print 'row[0]=', row[7]
			#print 'row[5]=', row[5]
			mydates.append(int(row[7]))  # row number, most recent is lowest number
			myprices.append(float(row[5])) # closing price for the day
			i=i+1
			if i==n:
				break
        #mydates.reverse()
        #myprices.reverse()
	return

	

def fit_lin_model():
	print 'fitting lin model'
	global mydates
	global myprices 
	mydates = np.reshape(mydates,(len(mydates), 1)) # converting to matrix of n X 1
	svr_lin.fit(mydates, myprices)

def fit_poly_model():
	print 'fitting poly model'
	global mydates
	global myprices
	mydates = np.reshape(mydates,(len(mydates), 1)) # converting to matrix of n X 1
	svr_poly.fit(mydates, myprices)

def fit_rbf_model():
	print 'fitting lin model'
	global mydates
	global myprices 
	mydates = np.reshape(mydates,(len(mydates), 1)) # converting to matrix of n X 1
	svr_rbf.fit(mydates, myprices)

def plot_lin_model():
	print 'plotting lin model'
	global mydates
	global myprices
	plt.scatter(mydates, myprices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(mydates,svr_lin.predict(mydates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

def plot_poly_model():
	print 'plotting poly model'
	global mydates
	global myprices
	plt.scatter(mydates, myprices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(mydates,svr_poly.predict(mydates), color= 'blue', label= 'Poly model') # plotting the line made by poly kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

def plot_rbf_model():
	print 'plotting lin model'
	global mydates
	global myprices
	plt.scatter(mydates, myprices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(mydates,svr_rbf.predict(mydates), color= 'red', label= 'RBF model') # plotting the line made by linear kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()


def predict_lin_model(x):
	print 'Linear model predicting value for day=', x 
	predicted_value = svr_lin.predict(x)[0]
	print 'Linear model predicted value=', predicted_value 

def predict_poly_model(x):
	print 'Poly model predicting value for day=', x 
	predicted_value = svr_poly.predict(x)[0]
	print 'Poly model predicted value=', predicted_value 

def predict_rbf_model(x):
	print 'RBF model predicting value for day=', x 
	predicted_value = svr_rbf.predict(x)[0]
	print 'RBF model predicted value#=', predicted_value 


def build_model(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma= 0.1) # defining the support vector regression models
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0] , svr_lin.predict(x)[0], svr_poly.predict(x)[0]
    #return svr_lin.predict(x)[0]

get_data('ung.csv',20) # calling get_data method by passing the csv file to it

print "Dates- ", mydates
print "Prices- ", myprices

# LINEAER MODEL
#fit_lin_model()
#plot_lin_model()
#predict_lin_model(0)


# POLY MODEL 
fit_poly_model()
plot_poly_model()
predict_poly_model(0)

# RBF MODEL 
fit_rbf_model()
plot_rbf_model()
predict_rbf_model(0)



