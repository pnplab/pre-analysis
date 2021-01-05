import multiprocessing as mp
import time


def foo_pool(x):
    time.sleep(5)
    return x*x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)



if __name__ == '__main__':
    pool = mp.Pool()
    for i in range(10):
        pool.apply_async(foo_pool, args=(i,), callback=log_result)
    pool.close()
    print(result_list)
