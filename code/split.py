import sys

f = open(sys.argv[1], 'r') #training set
essays = [x.strip('\n') for x in f.readlines()]

f1 = open(sys.argv[2], 'w') #training set
f2 = open(sys.argv[3], 'w') #validate

validation = []
train = []
count = 0
i = 1
sets = {}
for each in essays:
	set_id = each.split('\t')[1]
	l = sets.get(set_id, [])
	l.append(each)
	sets[set_id] = l

for s in sets.keys():
	validation.extend(sets[s][:len(sets[s])/4])
	train.extend(sets[s][len(sets[s])/4:])

f1.write('\n'.join(train))
f2.write('\n'.join(validation))

f1.close()
f2.close()

