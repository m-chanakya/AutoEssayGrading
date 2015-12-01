#! /usr/bin/python

import sys, math
from fileparse import get_list

def calc_mean(l):
	mean = float(sum(l))/len(l)
	return mean

def calc_std_dev(objects):

	l = []
	for each in objects:
		l.append(float(each.d1_score))
	
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

	
	def find_feature(self):
		
		max = -1
		feat = -1
		
		for i in xrange(len(self.subset[0].vector) - 1):
			temp = self.reduce_entropy(i)
			if temp > max:
				max = temp
				feat = i
		
		self.feature = feat

	
	def split(self):
		
		#CONDITION FOR NOT SPLITTING
		print self.std_dev
		if len(self.subset) <= self.MIN_SIZE:
			return

		if not self.feature:
			self.find_feature()

		if self.reduce_entropy(self.feature) < self.THRESHOLD:
			print self.reduce_entropy(self.feature)
			return

		total = len(self.subset)
		mean = 0
		for each in self.subset:
			mean += each.vector[self.feature]
		mean /= 1.0*total

		left = []
		right = []
		count = 0
		for each in self.subset:
			if each.vector[self.feature] <= mean:
				count += 1
				left.append(each)
			else:
				right.append(each)
		self.children = [Node(left), Node(right)]

	
	def make_tree(self):
	
		self.split()
		for each in self.children:
			each.make_tree()


	def predict(self, object):
		pass

	def printme(self):
		print self.subset
		print '*'*10
		for each in self.children:
			each.printme()
		


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage : python random_forest.py <train> <test>"
		sys.exit(1)
	root = Node (get_list('../data/rft'))
	root.make_tree()
	root.printme()
