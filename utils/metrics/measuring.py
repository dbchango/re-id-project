import tracemalloc


def tracing_start():
    """
    It starts trace
    :return: bool
    """
    tracemalloc.stop()
    tracemalloc.start()
    return tracemalloc.is_tracing()


def tracing_mem():
    """
    Will calculate memory (MB)
    :return: float
    """
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak size in MB - ", peak)
    return peak
