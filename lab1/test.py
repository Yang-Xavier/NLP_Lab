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
a = np.random.randint(0,10,(6,5))
# at = np.transpose(a)
# c = np.where(at[index]>2)

print(a)
a[:,[2,3]] =  a[:,[3,2]]
print(a)
# print(np.where)