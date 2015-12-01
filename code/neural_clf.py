#! /usr/bin/python

import sys
import math
import random

#GLOBALS
nof_hidden_units = 4
nof_op_units = 2
nof_ip_units = 64
eta = 1
threshold = 0.01

hidden = []
output = []

def activation(x):
	if (x<-5):
		x = -5
	ans = 1.0/(1.0 + math.e**(-1.0*x))
	return ans

def activation_derivative(x):
	temp = activation(x)
	return 1.0*temp*(1-temp)

def forward(inputs):
	z = []
	for a in xrange(len(inputs)):
		y = []
		for j in xrange(len(hidden)):
			s = 0
			for k in xrange(len(inputs[a])):
				s += 1.0*inputs[a][k]*hidden[j][k]
			y.append(activation(s))
		z.append([])
		for i in xrange(len(output)):
			s = 0
			for j in xrange(len(y)):
				s += 1.0*y[j]*output[i][j]
			z[a].append(activation(s))
	return z

def init_weights():
	global hidden, output
	for i in xrange(nof_hidden_units):
		temp = []
		for j in xrange(nof_ip_units):
			#wt = random.choice([-1, 1])
			#wt = (random.random()-0.5)*2*(1.0/math.sqrt(nof_hidden_units))
			wt = random.uniform(-0.1/math.sqrt(nof_hidden_units), 0.1/math.sqrt(nof_hidden_units))
			temp.append(wt)
		hidden.append(temp)

	for i in xrange(nof_op_units):
		temp = []
		for j in xrange(nof_hidden_units):
			wt = random.choice([-1, 1])
			#wt = (random.random()-0.5)*2*(1.0/math.sqrt(nof_hidden_units))
			#wt = random.uniform(-1.0/math.sqrt(nof_hidden_units), 1.0/math.sqrt(nof_hidden_units))
			temp.append(wt)
		output.append(temp)

def train(inputs, t):
	global hidden, output
	counter = 0
	while True and counter <= 10000:
		counter += 1
		#LEARN WEIGHTS
		for a in xrange(len(inputs)):
			#FORWARD
			y = []
			z = []
			netj = []
			netk = []
			old_hidden = hidden
			old_output = output
			for j in xrange(len(hidden)):
				s = 0
				for k in xrange(len(inputs[a])):
					s += 1.0*inputs[a][k]*hidden[j][k]
				netj.append(s)
				y.append(activation(s))
			
			for i in xrange(len(output)):
				s = 0
				for j in xrange(len(hidden)):
					s += 1.0*y[j]*output[i][j]
				netk.append(s)
				z.append(activation(s))

			#BACKWARD
			dk = []
			for k in xrange(len(output)):
				for j in xrange(len(hidden)):
					sensitivity = 1.0*(t[a][k]-z[k])*activation_derivative(netk[k])
					wkj = 1.0*eta*sensitivity*y[j]
					output[k][j] += wkj
				dk.append(sensitivity)

			for j in xrange(len(hidden)):
				sensitivity = 0
				for k in xrange(len(output)):
					sensitivity += 1.0*output[k][j]*dk[k]
				sensitivity *= 1.0*activation_derivative(netj[j])
				for i in xrange(len(inputs[a])):
					wji = 1.0*eta*inputs[a][i]*sensitivity 
					hidden[j][i] += wji

			flag = True
			for k in xrange(len(z)):
				if abs(t[a][k] - z[k]) > threshold:
					flag = False

			if flag:
				break
		if flag:
			break
	

def main():
	if len(sys.argv) != 2:
		print "Usage : python ann.py <data-file>"

	#TRAINING
	f = open(sys.argv[1], 'r')
	data = [each.strip('\n') for each in f.readlines()]
	#data = [each for each in data if (each.split(',')[-1] == '0' or each.split(',')[-1] == '7')]
	inputs = [ [float(i) for i in each.split(',')[:-1]] for each in data]
	t = [ [1, 0] if i == 0 else [0, 1] for i in [float(each.split(',')[-1]) for each in data] ]

	#NORMALIZE DATA
	for i in xrange(64):
		sum = 0
		for j in xrange(len(inputs)):
			sum += inputs[j][i]
		if sum:
			sum /= 1.0*len(inputs)
			variance = 0
			for j in xrange(len(inputs)):
				variance += (sum - inputs[j][i])**2
			variance /= 1.0*len(inputs)
			for j in xrange(len(inputs)):
				inputs[j][i] = 1.0*(inputs[j][i] - sum)/variance

	init_weights()
	train(inputs, t)
	print hidden
	print output

	#TESTING
	f = open(sys.argv[2], 'r')
	data = [each.strip('\n') for each in f.readlines()]
	#data = [each for each in data if (each.split(',')[-1] == '0' or each.split(',')[-1] == '7')]
	inputs = [ [float(i) for i in each.split(',')[:-1]] for each in data]
	t = [ [1, 0] if i == 0 else [0, 1] for i in [float(each.split(',')[-1]) for each in data] ]

	#NORMALIZE DATA
	for i in xrange(64):
		sum = 0
		for j in xrange(len(inputs)):
			sum += inputs[j][i]
		if sum:
			sum /= 1.0*len(inputs)
			variance = 0
			for j in xrange(len(inputs)):
				variance += (sum - inputs[j][i])**2
			variance /= 1.0*len(inputs)
			for j in xrange(len(inputs)):
				inputs[j][i] = 1.0*(inputs[j][i] - sum)/variance
	
	z = forward(inputs)
	# print accuracy(z, t)
	

if __name__ == "__main__":
	main()