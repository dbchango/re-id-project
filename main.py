# from multiprocessing import Pool
# import torch.multiprocessing as mp

# def f(x):
#      return x*x

# if __name__ == '__main__':
#     num_processes = 4
#     processes = []
#     for range in range(num_processes):
#          p = mp.Process(target=f, args=(4,))

#          p.start()
#          processes.append(p)
#     for p in processes:
#          p.join()

# importing the multiprocessing module
import multiprocessing
import torch.multiprocessing as mp
import os
import time
def worker1():
    # printing process id
    print("ID of process running worker1: {}".format(os.getpid()))
  
def worker2():
    # printing process id
    print("ID of process running worker2: {}".format(os.getpid()))
  
if __name__ == "__main__":
     i = time.time()
     # printing main program process id
     print("ID of main process: {}".format(os.getpid()))
     p1 = mp.Process(target=worker1)
     p2 = mp.Process(target=worker2)
     # creating processes
     # p1 = multiprocessing.Process(target=worker1)
     # p2 = multiprocessing.Process(target=worker2)

     # starting processes
     p1.start()
     p2.start()

     # process IDs
     print("ID of process p1: {}".format(p1.pid))
     print("ID of process p2: {}".format(p2.pid))

     # wait until processes are finished
     p1.join()
     p2.join()
     f = time.time()
     # both processes finished
     print("Both processes finished execution!")
     print("Time: {}")

     # check if processes are alive
     print("Process p1 is alive: {}".format(p1.is_alive()))
     print("Process p2 is alive: {}".format(p2.is_alive()))