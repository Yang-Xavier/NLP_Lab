# from scipy.sparse import coo_matrix
# #
# # a = [0,1,2,3,4,0]
# # b = [0,1,2,3,4,0]
# # c = [1,1,1,1,1,1]
# #
# # d = coo_matrix((c,(a,b)))
# # print(d.toarray())

import multiprocessing as mp
import time,os,random

def long_time_task(name,e):     #执行函数
    s = []
    for i in range(10000000):
        s.append(e)
        if i%1000 ==0:
            print(i)
    return  s

def test():
    a = []

    def c(s):
        a.extend(s[:])

    p = mp.Pool(4)#意为设定同时进行的子进程数,在一个进程池内,根据cpu核数进行的.      
    for i in range(400000000):#为8个进程提供name 0,1,2,3,4,5,6,7
        p.apply_async(long_time_task, (i,2), callback=c)#apply_async()方法下面说明
        # a.append(2)
    p.close() #关闭进程池,不在接收新的任务
    p.join()
    print(len(a))


t = time.clock()
test()
print("%.2f"%(time.clock()-t))
