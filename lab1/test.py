# from collections import *
# import re
# a = "I'm a super hero"
#
# r= Counter(re.sub("[^\w]"," ",a).split())
# print(r)

# import pandas as pd
#
# a = {"a":1,"b":2,"c":3}
# b = {"a":1,"b":2}
# k = ["a","b","c"]
#
# data = pd.DataFrame(columns = k, data = [a,b], dtype=float)
# data.fillna(0,inplace=True)
# print(data)

import numpy as np
# np.random.seed(100000)
#
# index = 3
#
# a = np.random.randint(0,10,(5,5))
# at = np.transpose(a)
# c = np.where(at[index]>2)
a = np.array([1,2,3,4])
print(np.where(a>0)[0][:10])
print(np.append(np.array([1,2,3,4]), np.array([2,3,4,5])))