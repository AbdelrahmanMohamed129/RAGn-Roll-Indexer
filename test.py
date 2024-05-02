# write any parallelizable code
import time
import random
from multiprocessing import Pool, cpu_count

def worker(n):
    print(n)
    time.sleep(random.randint(1, 5))
    return n

def main():
    # get the number of CPU cores
    with Pool(cpu_count()) as p:
        result = p.map(worker, range(50))
    print(result)
    
if __name__ == '__main__':
    main()