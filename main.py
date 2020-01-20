import sys
import json
import numpy as np
import numpy.linalg as linalg

# The expected command is of the form 
# python3 main.py data/file_name.in data/file_name.json

# Read in the arguments given on the command line
args = sys.argv

# Read in data from data/filename.in (aka args[1])
data = np.loadtxt(args[1])

# Read in hyperparamaters from data/filename.json (aka args[2])
with open(args[2]) as json_file:
    hyper = json.load(json_file)

# lls_analyic analyticly  determines the weights of a problem given:
# x: An array containing all the samples of x
# y: An array containing y
# learn_rate: The learning rate
# num_itr: The number of iterations
# Returns a vector containing the calculated weights
def lls_analytic(x,y):

    # we must add the bias parameter into our x matrix
    bias = np.ones((np.shape(x)[0],1))
    x = np.hstack((bias,x))

    # calculate the transpose of x
    x_t = np.matrix.transpose(x)

    # calculate the moore-penrose inverse
    pm = np.matmul(linalg.inv(np.matmul(x_t,x)),x_t)

    # calculate and return the weights
    return np.matmul(pm,y)

# lls_gd uses stochastic gradient descent to determine the weights of a problem given:
# x: An array containing all the samples of x
# y: An array containing y
# learn_rate: The learning rate
# num_itr: The number of iterations
# Returns a vector containing the calculated weights
def lls_gd(x,y,learn_rate,num_itr):

    # we must add the bias parameter into our x matrix
    bias = np.ones((x.shape[0],1))
    x = np.concatenate((bias,x),axis=1)

    # Initalize the weights
    w = np.ones(np.shape(x)[1])

    #iterate num_itr times
    for i in range(num_itr):

        for n in range(x.shape[0]):
            w += learn_rate * (y[n] - np.dot(w, x[n])) * x[n]

    return w

# Calculate the weights
w_analytic = lls_analytic(data[:,0:-1], data[:,-1])
w_gd = lls_gd(data[:,0:-1], data[:,-1],hyper["learning rate"],hyper["num iter"])

# Properly shape arrays for output
w_analytic.transpose()
w_gd.transpose()

# Extract file name from arguments
fname = args[1][0:-3] + ".out"

# Create our output file
f = open(fname,"w+")

# Output to files
for w in w_analytic:

    f.write("{0:.4f}\n".format(w))

# Seperate w_analytic and w_gd by a line
f.write("\n")

for w in w_gd:

    f.write("{0:.4f}\n".format(w))

# Save file
f.close()






