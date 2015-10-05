class files:
	def __init__(self, arr):
		self.essay_id = arr[0]
		self.essay_set = arr[1]
		self.essay = arr[2]
		self.r1_d1 = arr[3]
		self.r2_d1 = arr[4]
		self.r3_d1 = arr[5]
		self.d1_score = arr[6]
		self.r1_d2 = arr[7]
		self.r2_d2 = arr[8]
		self.d2_score = arr[9]
		self.r1_t1 = arr[10]
		self.r1_t2 = arr[11]
		self.r1_t3 = arr[12]
		self.r1_t4 = arr[13]
		self.r1_t5 = arr[14]
		self.r1_t6 = arr[15]
		self.r2_t1 = arr[16]
		self.r2_t2 = arr[17]
		self.r2_t3 = arr[18]
		self.r2_t4 = arr[19]
		self.r2_t5 = arr[20]
		self.r2_t6 = arr[21]
		self.r3_t1 = arr[22]
		self.r3_t2 = arr[23]
		self.r3_t3 = arr[24]
		self.r3_t4 = arr[25]
		self.r3_t5 = arr[26]
		self.r3_t6 = arr[27]


def get_list():
	fo = open('../data/training_set.tsv')
	lines = fo.readlines()
	fo.close()

	obj_arr = []
	for each in lines:
		send_obj = files(each.split('\n')[0].split('\t'))
		obj_arr.append(send_obj)

	return obj_arr
