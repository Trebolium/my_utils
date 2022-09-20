import concurrent.futures
import time, datetime, pdb
from tqdm import tqdm


""" Broadcasts args to the list and submits this to the function
    Execute in multithreaded context
""" 
def multithread_chunks(multithread_function, list_for_chunking, num_threads, args):

    iterables = [[entry] + args for entry in list_for_chunking]
    start_time = time.time()
    exceptions = []
    for i in tqdm(range(0, len(iterables), num_threads)):
        
        et = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
        print(f'Elapsed: {et}: Progress {i}/{len(iterables)}')

        chunked_iterables = iterables[i:i+num_threads]
        # pdb.set_trace()
        # multithread_function(chunked_iterables[0])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_objects = executor.map(multithread_function, chunked_iterables)
