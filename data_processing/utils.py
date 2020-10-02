def cycle_no_memory(iterable):
    '''Copy of cycle from itertools but with no memory'''
    while True:
        for element in iterable:
            yield element

def cycle_n_times(n, iterable):
    iterate = cycle_no_memory(iterable)
    for i in range(n):
        yield next(iterate)

if __name__ == "__main__":
    hi = [1,2,3]
    for elt in cycle_n_times(10, hi):
        print(elt)
