import numpy as np
import pickle

fd = open('data/segmentation.test.txt', 'rb')
content = fd.readlines()
content = content[5:] #ignore the first 5 lines

def custon_set(array):
    ret_set = []
    for a in array:
        if not a in ret_set:
            ret_set.append(a)
    return ret_set

elems = []
for l in content:
    s = l.split(',')[0]
    elems.append(s)

set_tags = custon_set(elems)
set_dict = {}
index = 0
for t in set_tags:
    set_dict[t] = index
    index += 1

U = np.zeros((len(content), len(set_tags)))
index = 0
for e in elems:
    U[index][set_dict[e]] = 1
    index += 1

U_out = open('result_pickles/apriori_U.pickle', 'wb')
pickle.dump(U, U_out)
U_out.close()
