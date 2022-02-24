from contextlib import contextmanager
import sys, os
import time
# import matplotlib.pyplot as plt
import numpy as np
import pickle
def timing(func):
    def inner(*args, **kwargs):
        st_time = time.time()
        ret = func(*args, **kwargs)
        ed_time = time.time()
        print(f'[{func.__name__}] uses {ed_time-st_time:.1f} secs')
        return ret
    return inner

@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()
    fd2 = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        os.dup2(to.fileno(), fd2) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        sys.stderr = os.fdopen(fd2, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def plot_trace(perf_time_trace, outdir, tag=''):
    perf_time_trace = np.array(perf_time_trace)
    latencys = -perf_time_trace[:,0]
    energys = -perf_time_trace[:, 1]
    runtimes = perf_time_trace[:, 2]
    samples = perf_time_trace[:, 3]
    fig = plt.figure()
    plt.plot(runtimes, latencys)
    plt.xlabel('secs')
    plt.ylabel('Perf (latency) (cycles)')
    runtime_graph_path = os.path.join(outdir, f'perf_2_runtime{tag}.jpg')
    plt.savefig(runtime_graph_path)
    plt.close()
    fig = plt.figure()
    plt.plot(samples, latencys)
    plt.xlabel('Samples')
    plt.ylabel('Perf (latency) (cycles)')
    sample_graph_path = os.path.join(outdir, f'perf_2_samples{tag}.jpg')
    plt.savefig(sample_graph_path)
    plt.close()
    chkpt = {'gamma': perf_time_trace}
    # with open(os.path.join(outdir, 'gamma_perf_trace.plt'), 'wb') as fd:
    #     pickle.dump(chkpt, fd)

import numpy as np


# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto(fitnesses, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(fitnesses.shape[0])
    n_points = fitnesses.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(fitnesses):
        nondominated_point_mask = np.any(fitnesses>fitnesses[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        fitnesses = fitnesses[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask, len(is_efficient)
    else:
        return is_efficient