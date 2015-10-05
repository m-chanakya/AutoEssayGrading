import token_gen
import nltk
import enchant

Dict = enchant.Dict("en_US")

class files:

	def __init__(self, arr):
		"""
		Initiliases an object out of given features
		===========================================
		Parameters :
			* array
				- contains set of features
		===========================================
		"""
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

		self.token_arr, self.punc_arr = token_gen.tokenize([self.essay])

	def set_word_count(self, min_len):
		"""
		Sets different count (of words) / spelling related features
		================================================
		Sets the following:
			* sets the word count in total
			* sets the long word count
			* sets the average word lenth
			* sets the spelling errors
		================================================
		"""

		self.word_count = len(self.token_arr)

		self.dic = {}
		self.long_word_count = 0
		self.avg_word_len = 0
		self.lex_diversity = 0
		self.spell_errors = 0

		for each in self.token_arr:
			if len(each) > min_len:
				self.long_word_count += 1
			if not self.dic.has_key(each):
				self.dic[each] = 1
				self.lex_diversity += 1
			if len(each) > 1:
				try:
					if not Dict.check(each.encode('utf8')):
						self.spell_errors += 1
				except:
					self.spell_errors += 1
			self.avg_word_len += len(each)

		self.avg_word_len = float(self.avg_word_len) / float(self.word_count)

	def set_pos_features(self):
		"""
		Sets tag related features
		===============================================
		Sets different features like:
			* Nount count
			* Verb count
			* Adjective count
			* Adverb count
		===============================================
		"""

		self.pos_tags = nltk.pos_tag(self.token_arr)
		
		self.verb_count = 0
		self.noun_count = 0
		self.adj_count = 0
		self.adv_count = 0

		for each in self.pos_tags:
			if each[1].startswith('NN'):
				self.noun_count += 1
			elif each[1].startswith('JJ'):
				self.adj_count += 1
			elif each[1].startswith('RB'):
				self.adv_count += 1
			elif each[1].startswith('VB'):
				self.verb_count += 1


	def set_punctuation_features(self):
		"""
		Sets different featured realted to punctuations
		===============================================
		Sets different features like:
			* Comma count
			* Quotation mark count
			* Punctuation count
			* Sentence count
		===============================================
		"""

		self.comma_count = 0
		self.sen_count = 0
		self.punc_count = 0
		self.quo_count = 0

		for each in self.punc_arr:
			self.punc_count += 1
			if each == ',':
				self.comma_count += 1
			elif each == '.':
				self.sen_count += 1
			elif each == '"':
				self.quo_count += 1

	def set_vectors(self):
		"""
		Initalises the vector of an object
		"""

		self.vector = [self.word_count, self.long_word_count, self.avg_word_len, \
				self.lex_diversity, self.spell_errors, self.verb_count, \
				self.noun_count, self.adj_count, self.adv_count, \
				self.comma_count, self.sen_count, self.punc_count, \
				self.quo_count, float(self.d1_score)]


def get_list():
	"""
	Creates an array of objects out of 
	input training file
	==================================
	Returns:
		* array of objects where each
		object corresponds to a document
	==================================
	"""

	fo = open('../data/code_test.tsv')
	lines = fo.readlines()
	fo.close()

	obj_arr = []
	vec_arr = []
	for each in lines:
		send_obj = files(each.split('\n')[0].split('\t'))
		send_obj.set_word_count(5)
		send_obj.set_pos_features()
		send_obj.set_pos_features()
		send_obj.set_punctuation_features()
		send_obj.set_vectors()
		obj_arr.append(send_obj)

	return obj_arr

if __name__ == "__main__":
	arr = get_list()
	print arr[0].pos_tags
	print arr[0].verb_count
	print arr[0].vector
