from fileparse import get_list
from text_rank import get_only_phrases
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
def create_top_phrases(filename):
	docx = get_list(filename)
	arr = []
	dic = {}
	for each in docx:
        	keys = each.phrases
		for every in keys:
			if dic.has_key(every):
				dic[every] += 1
			else:
				dic[every] = 1
	print dic
create_top_phrases('../data/phrase_only_data.csv')	
