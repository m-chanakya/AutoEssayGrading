#! /usr/bin/python

import sys, math
from fileparse import get_list
from random import shuffle
from metrics import get_average_kappa as kappa

def calc_mean(l):
	mean = float(sum(l))/len(l)
	return mean

def calc_std_dev(objects):

	l = []
	for each in objects:
		l.append(each.vector[-1])
	
	if not l:
		return 0	
	
	mean = calc_mean(l)
	ans = 0
	
	for each in l:
		ans += (each - mean)**2
	ans /= 1.0*len(l)
	
	return math.sqrt(ans)

class Node:
	
	THRESHOLD = 0.1
	MIN_SIZE = 2


	def __init__ (self, objects):
		
		self.std_dev = calc_std_dev(objects)
		self.subset = objects
		self.children = []
		self.feature = None
		self.mean = None


	def reduce_entropy(self, feat):
		
		total = len(self.subset)
		count = 0
		left = []
		right = []

		mean = 0
		for each in self.subset:
			mean += each.vector[feat]
		mean /= 1.0*total

		for each in self.subset:
			if each.vector[feat] <= mean:
				count += 1
				left.append(each)
			else:
				right.append(each)

		temp = self.std_dev - \
				(1.0*count/total * calc_std_dev(left) +\
				1.0*(total-count)/total * calc_std_dev(right))

		return temp, mean

	
	def find_feature(self):
		
		max = -1
		feat = -1
		mean = -1
		
		for i in xrange(len(self.subset[0].vector) - 1):
			temp, m = self.reduce_entropy(i)
			if temp > max:
				max = temp
				feat = i
				mean = m
		
		self.feature = feat
		self.mean = mean

	
	def split(self):
		#CONDITION FOR NOT SPLITTING
		if len(self.subset) <= self.MIN_SIZE:
			return 0

		if not self.feature:
			self.find_feature()

		if self.reduce_entropy(self.feature) < self.THRESHOLD:
			return 0

		left = []
		right = []
		count = 0
		for each in self.subset:
			if each.vector[self.feature] <= self.mean:
				count += 1
				left.append(each)
			else:
				right.append(each)
		self.children = [Node(left), Node(right)]

		return 1


def print_list(l):

	for each in l:
		print each.d1_score,
	print

def printme(root):

	print root.feature
	if root.children == []:
		print_list(root.subset)
		print '*'*10
	for each in root.children:
		printme(each)


def make_tree(root):
	
	if root.split():
		for each in root.children:
			make_tree(each)


def predict(root, obj):

	if root.children == []: #leaf
		sum = 0
		for each in root.subset:
			sum += each.vector[-1]
		return 1.0*sum/len(root.subset)

	if obj.vector[root.feature] <= root.mean:
		return predict(root.children[0], obj)
	else:
		return predict(root.children[1], obj)

		
def decision_tree(train, test):

	#TRAINING
	tree = Node(train)
	make_tree(tree)

	#TESTING
	predictions = []
	for each in test:
		print each.vector[-1]
		predictions.append(predict(tree, each))
	return predictions


def random_forest(train, test):
	
	SIZE = (3*len(train))/4
	NO_OF_TREES = 5
	
	#TRAINING
	trees = []
	for i in xrange(NO_OF_TREES):
		shuffle(train)
		root = Node (train[:SIZE])
		make_tree(root)
		trees.append(root)

	#TESTING
	predictions = []
	for each in test:
		sum = 0
		for tree in trees:
			temp = predict(tree, each)
			sum += temp
		sum /= 1.0*NO_OF_TREES
		predictions.append(sum)

	return predictions


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "Usage : python random_forest.py -t|-m <train>|<model> <test>"
		sys.exit(1)

	print "\nInitializing Random Forest..."
	#READ DATA
	if sys.argv[1] == '-t':
		train = get_list(sys.argv[2], True)
	elif sys.argv[1] == '-m':
		train = get_list(sys.argv[2], False)
	else:
		print "Usage: python neural_net.py -t|-m <train> <test>"
		sys.exit(1)
	print "Ready.\n"
	print "Reading test data..."
	test = get_list(sys.argv[3])
	answers = [each.vector[-1] for each in test]
	desc_ans = decision_tree(train, test)
	random_forest_ans = random_forest(train, test)

	kp = kappa(answers, desc_ans)
	print "\nThe Average Quadratic Weighted Kappa obtained for Decision tree is: ", kp, "\n"
	print "="*50

	kp = kappa(answers, random_forest_ans)
	print "\nThe Average Quadratic Weighted Kappa obtained for Random Forest is: ", kp, "\n"
	print "="*50