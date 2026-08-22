import time
from contextlib import contextmanager


def timed(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"{function.__name__} took {duration:.6f} seconds")
        return result

    return wrapper


@contextmanager
def open_data(path):
    file = open(path, "r")
    try:
        yield file
    finally:
        file.close()


@timed
def longest_word(text):
    words = text.split()
    return max(words, key=len)