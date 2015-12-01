import fileparse
def create_file():
    fp = open('../data/neural_data.csv', 'a')
    filename = '../data/SMALL_TRAIN'
    docs_list = fileparse.get_list(filename)
    for doc in docs_list:
        arr = doc.vector
        string = ''
        for i in xrange(0, len(arr) - 1):
            string += str(arr[i])
            string += ', '
        string += str(arr[-1])
        string += '\n'
        fp.write(string)
    fp.close()

create_file()
