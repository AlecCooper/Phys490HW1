import sys
import json
import numpy as np

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
def lls_analytic(x,y,learn_rate,num_itr):

    return ""

# lls_gd uses stochastic gradient descent to determine the weights of a problem given:
# x: An array containing all the samples of x
# y: An array containing y
# learn_rate: The learning rate
# num_itr: The number of iterations
# Returns a vector containing the calculated weights
def lls_gd(x,y,learn_rate,num_itr):

    return ""




# Extract fname from arguments
fname = args[1][0:-3]







