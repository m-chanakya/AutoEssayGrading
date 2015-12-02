#! /usr/bin/python

import sys, math, random
from fileparse import get_list
import pickle
from metrics import get_average_kappa as kappa

#GLOBALS
nof_hidden_units = 4
nof_op_units = 1
nof_ip_units = 13
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
	init_weights()
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
	
def normalize(data, means, variances):
	for i in xrange(nof_ip_units):
		for j in xrange(len(data)):
			if variances[i]:
				data[j][i] = 1.0*(data[j][i] - means[i])/variances[i]

def preprocess(train):
	means = []
	variances = []
	for i in xrange(nof_ip_units):
		sum = 0
		for j in xrange(len(train)):
			sum += train[j][i]
		if sum:
			sum /= 1.0*len(train)
			means.append(sum)
			variance = 0
			for j in xrange(len(train)):
				variance += (sum - train[j][i])**2
			variance /= 1.0*len(train)
			variance = math.sqrt(variance)
			variances.append(variance)
		else:
			means.append(0)
			variances.append(0)
	# print means
	# print variances
	return means, variances

def main():
	if len(sys.argv)<3:
		print "Usage: python neural_net.py -t|-m <train>|<model> <test>"
		sys.exit(1)

	print "\nInitializing Neural Net..."
	#READ DATA
	if sys.argv[1] == '-t':
		raw = get_list(sys.argv[2], True)
	elif sys.argv[1] == '-m':
		raw = get_list(sys.argv[2], False)
	else:
		print "Usage: python neural_net.py -t|-m <train>|<model> <test>"
		sys.exit(1)

	train_data = [each.vector[:-1] for each in raw]
	train_ans = [each.vector[-1:] for each in raw]
	
	print "Ready.\n"
	print "Reading test data..."
	raw = get_list(sys.argv[3])
	test_data = [each.vector[:-1] for each in raw]
	test_ans = [each.vector[-1:] for each in raw]
	answers = [each.vector[-1] for each in raw]

	global nof_ip_units
	nof_ip_units = len(test_data[0])

	#NORMALIZE DATA
	means, variances = preprocess(train_data)
	normalize(train_data, means, variances)
	normalize(test_data, means, variances)
	
	#TRAIN
	train(train_data, train_ans)

	#PREDICT
	predictions = forward(test_data)
	predictions = [each[0] for each in predictions]
	kp = kappa(answers, predictions)
	print answers
	print predictions
	print "\nThe Average Quadratic Weighted Kappa obtained is: ", kp, "\n"
	print "="*50
	# print test_ans
	# print predictions

if __name__ == "__main__":
	main()