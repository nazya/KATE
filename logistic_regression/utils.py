#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.datasets import load_svmlight_file

def prepare_data(filename):
    data = load_svmlight_file("datasets/" + filename + ".txt")
    A, y = data[0], data[1]
    m,n = A.shape
    
    if (2 in y) & (1 in y):
        y = 2 * y - 3
    if (2 in y) & (4 in y):
        y = y - 3
    if (1 in y) & (0 in y):
        y = 2*y - 1
    assert((-1 in y) & (1 in y))
    
    return A.toarray(), y, m, n