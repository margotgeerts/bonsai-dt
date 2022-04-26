#cython: boundscheck=False
#cython: wrapround=False
#cython: cdivision=True

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan
from libc.math cimport sqrt, pow
from libc.stdio cimport printf
from libcpp cimport bool


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef unsigned long ULong


cpdef DTYPE_t square(DTYPE_t x) nogil: 
    return x * x

cpdef double euclidean(double x10, double x11, double x20, double x21) nogil: 
    return sqrt(square(x10 - x20) + square(x11 - x21))

cpdef DTYPE_t mse(DTYPE_t [:] y, DTYPE_t y_hat) nogil:
    cdef double diffsq = 0
    cdef int arr_shape = y.shape[0]
    cdef int k

    for k in range(arr_shape):
        diffsq += square((y[k] - y_hat))
    return diffsq / arr_shape

cpdef DTYPE_t mean(DTYPE_t [:] y) nogil:
    cdef double sum_y = 0
    cdef int arr_shape = y.shape[0]

    for k in range(arr_shape):
        sum_y += y[k]
    return sum_y / arr_shape


def reorder(X, y, z, i_start, i_end, j_split, split_value, missing):
    return _reorder(X, y, z, i_start, i_end, j_split, split_value, missing)
 

cdef size_t _reorder(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        size_t i_start, 
        size_t i_end, 
        np.ndarray[np.int32_t, ndim=1] j_split, 
        np.ndarray[DTYPE_t, ndim=1] split_value, 
        np.ndarray[np.int32_t, ndim=1] missing):
    """
    - X: 2-d numpy array (n x m)
    - y: 1-d numpy array (n)
    - z: 1-d numpy array (n)
    - i_start: row index to start
    - i_end: row index to end
    - j_split: column index for the splitting variable
    - split_value: threshold
    """
    cdef size_t j
    cdef size_t m = X.shape[1]
    cdef size_t i_head = i_start
    cdef size_t i_tail = i_end - 1
    cdef size_t do_swap = 0
    cdef DTYPE_t intercept, slope, dist, dist_1, dist_2
    cdef DTYPE_t focal1_x, focal1_y, focal2_x, focal2_y

    with nogil:
        if split_value[2] != -1:
            focal1_x = X[<size_t>split_value[0], 0]
            focal1_y = X[<size_t>split_value[0], 1]
            focal2_x = X[<size_t>split_value[1], 0]
            focal2_y = X[<size_t>split_value[1], 1]
        while i_head <= i_tail:

            if i_tail == 0: 
                # if tail is 'zero', should break
                # otherwise, segmentation fault, 
                # as size_t has no sign. 0 - 1 => huge number
                break

            do_swap = 0 
            if split_value[1] == -1:
                if isnan(X[i_head,<size_t>j_split[0]]):
                    if missing[0] == 1: # send the missing to the right node
                        do_swap = 1
                else:
                    if X[i_head,<size_t>j_split[0]] >= <DTYPE_t>split_value[0]:
                        do_swap = 1
            else:
                if isnan(X[i_head,0]) | isnan(X[i_head,1]):
                    if missing[0] == 1: # send the missing to the right node
                        do_swap = 1
                else:
                    if split_value[2] == -1:
                        intercept, slope = split_value[0], split_value[1]
                        if slope * X[i_head, 0] + intercept >= X[i_head, 1]:
                            do_swap = 1
                    else:
                        dist = split_value[2]
                        dist_1 = sqrt(square(X[i_head, j_split[0]] - focal1_x) + square(X[i_head, j_split[1]] - focal1_y))
                        dist_2 = sqrt(square(X[i_head, j_split[0]] - focal2_x) + square(X[i_head, j_split[1]] - focal2_y))
                        if (dist_1 + dist_2) >= dist:
                            do_swap = 1
            

            if do_swap == 1:
                # swap X rows
                for j in range(m):
                    X[i_head,j], X[i_tail,j] = X[i_tail,j], X[i_head,j]
                # swap y, z values
                y[i_head], y[i_tail] = y[i_tail], y[i_head]
                z[i_head], z[i_tail] = z[i_tail], z[i_head]
                # decrease the tail index
                i_tail -= 1
            else:
                # increase the head index
                i_head += 1
        
        
        
    return i_head


def sketch(np.ndarray[DTYPE_t, ndim=2] X not None, 
        np.ndarray[DTYPE_t, ndim=1] y not None, 
        np.ndarray[DTYPE_t, ndim=1] z not None, 
        np.ndarray[DTYPE_t, ndim=2] xdim not None, 
        np.ndarray[DTYPE_t, ndim=3] cnvs not None, 
        np.ndarray[DTYPE_t, ndim=2] cnvsn not None,
          int use_mse):

    # canvas --> (sketch) --> avc 
    # AVC: Attribute-Value Class group in RainForest
    _sketch(X, y, z, xdim, cnvs, cnvsn, use_mse)
    return 0

def sketch_diagonal(np.ndarray[DTYPE_t, ndim=2] X not None, 
        np.ndarray[DTYPE_t, ndim=1] y not None, 
        np.ndarray[DTYPE_t, ndim=1] z not None, 
        np.ndarray[DTYPE_t, ndim=2] xdim not None, 
        np.ndarray[DTYPE_t, ndim=3] cnvs not None,
        int use_mse):

    # canvas --> (sketch) --> avc 
    # AVC: Attribute-Value Class group in RainForest
    _sketch_diagonal(X, y, z, xdim, cnvs, use_mse)
    return 0

def sketch_gaussian(np.ndarray[DTYPE_t, ndim=2] X not None, 
        np.ndarray[DTYPE_t, ndim=1] y not None, 
        np.ndarray[DTYPE_t, ndim=1] z not None, 
        np.ndarray[DTYPE_t, ndim=2] xdim not None, 
        np.ndarray[DTYPE_t, ndim=3] cnvs not None,
        size_t i_start,
        size_t i_end,
        int use_mse):

    # canvas --> (sketch) --> avc 
    # AVC: Attribute-Value Class group in RainForest
    _sketch_gaussian(X, y, z, xdim, cnvs, i_start, i_end, use_mse)
    return 0

cdef void _sketch(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        np.ndarray[DTYPE_t, ndim=2] xdim, 
        np.ndarray[DTYPE_t, ndim=3] cnvs, 
        np.ndarray[DTYPE_t, ndim=2] cnvsn,
        int use_mse):

    cdef size_t i, j, k, k_raw, k_tld
    cdef size_t n = X.shape[0]
    cdef size_t m = X.shape[1]
    cdef size_t n_cnvs = <size_t> cnvs.shape[0]/2
    cdef size_t n_bin
    cdef size_t xdim0 = <size_t> xdim[0, 4]
    cdef double k_prox
    cdef double y_i, z_i
    cdef double y_tot = 0.0
    cdef double z_tot = 0.0
    cdef double n_na, y_na, z_na
    cdef DTYPE_t [:] y_l = np.zeros(n)
    cdef DTYPE_t [:] y_r = np.zeros(n)

    # update E[y] & E[z]
    with nogil:

        for i in range(n):

            y_i = y[i]
            z_i = z[i]
            y_tot += y_i
            z_tot += z_i

            for j in range(m):

                #if xdim[j, 2] < 1e-12:
                #    continue
                if isnan(X[i, j]):
                    cnvsn[j, 1] += 1
                    cnvsn[j, 2] += y_i
                    cnvsn[j, 3] += z_i
                else:
                    k_prox = (X[i, j] - xdim[j, 1])/xdim[j, 2]
                    if k_prox < 0:
                        k_prox = 0
                    elif k_prox > xdim[j, 3] - 1:
                        k_prox = xdim[j, 3] - 1
                    k = <size_t> (k_prox + (xdim[j, 4] - xdim0)*2)
                    cnvs[k, 3, 0] += 1
                    cnvs[k, 4, 0] += y_i
                    cnvs[k, 5, 0] += z_i
                

        # accumulate stats
        for j in range(m):
            n_bin = <size_t> xdim[j, 3]
            
            for k_raw in range(1, n_bin): 
                k = <size_t> (k_raw + (xdim[j, 4] - xdim0)*2)
                cnvs[k, 3, 0] += cnvs[k-1, 3, 0] 
                cnvs[k, 4, 0] += cnvs[k-1, 4, 0] 
                cnvs[k, 5, 0] += cnvs[k-1, 5, 0] 
                # fill the right node at the same time
                cnvs[k, 6, 0] = n - cnvs[k, 3, 0] - cnvsn[j, 1]
                cnvs[k, 7, 0] = y_tot - cnvs[k, 4, 0] - cnvsn[j, 2]
                cnvs[k, 8, 0] = z_tot - cnvs[k, 5, 0] - cnvsn[j, 3]

            # fill the right node
            k = <size_t> ((xdim[j, 4] - xdim0)*2)
            cnvs[k, 6, 0] = n - cnvs[k, 3, 0] - cnvsn[j, 1]
            cnvs[k, 7, 0] = y_tot - cnvs[k, 4, 0] - cnvsn[j, 2]
            cnvs[k, 8, 0] = z_tot - cnvs[k, 5, 0] - cnvsn[j, 3]

        # missing values
        for j in range(m):

            n_bin = <size_t> xdim[j, 3]
            n_na = cnvsn[j, 1]
            y_na = cnvsn[j, 2]
            z_na = cnvsn[j, 3]

            if n_na == 0:
                continue

            for k_raw in range(n_bin):
                k = <size_t> (k_raw + (xdim[j, 4] - xdim0)*2)
                k_tld = k + n_bin
                cnvs[k_tld, 3, 0] = cnvs[k, 3, 0]
                cnvs[k_tld, 4, 0] = cnvs[k, 4, 0]
                cnvs[k_tld, 5, 0] = cnvs[k, 5, 0]
                cnvs[k_tld, 6, 0] = cnvs[k, 6, 0]
                cnvs[k_tld, 7, 0] = cnvs[k, 7, 0]
                cnvs[k_tld, 8, 0] = cnvs[k, 8, 0]
                cnvs[k_tld, 9, 0] = 1

                cnvs[k, 3, 0] += n_na
                cnvs[k, 4, 0] += y_na
                cnvs[k, 5, 0] += z_na
                cnvs[k_tld, 6, 0] += n_na
                cnvs[k_tld, 7, 0] += y_na
                cnvs[k_tld, 8, 0] += z_na
                
        if use_mse==1:
            for k in range(n_cnvs):
                for i in range(n):
                    y_i = y[i]
                    if X[i, <size_t>cnvs[k, 1,0]] < <DTYPE_t>cnvs[k,2,0]:
                        y_l[i] = y_i
                    else:
                        y_r[i] = y_i
                cnvs[k, 5, 0] = mse(y_l, mean(y_l))
                cnvs[k, 8, 0] = mse(y_r, mean(y_r))

    # done _sketch
                    
cdef void _sketch_diagonal(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        np.ndarray[DTYPE_t, ndim=2] xdim, 
        np.ndarray[DTYPE_t, ndim=3] cnvs,
        int use_mse):

    cdef size_t i, j, k
    cdef size_t n = X.shape[0]
    cdef size_t n_cnvs = <size_t> cnvs.shape[0]
    cdef double y_i, z_i
    cdef double x_i_0, x_i_1, x_j_0, x_j_1
    cdef DTYPE_t m1, m2, x2, y2, b
    cdef size_t idx = 0
    cdef DTYPE_t [:] y_l = np.zeros(n)
    cdef DTYPE_t [:] y_r = np.zeros(n)
    
    
    with nogil:
        for i in range(n):
            x_i_0 = X[i,0]
            x_i_1 = X[i,1]
            
            
            for j in range(i+1, n):
                x_j_0 = X[j,0]
                x_j_1 = X[j,1]
                
                m1 = (x_j_1-x_i_1)/(x_j_0-x_i_0)
                m2 = -1 * pow(m1, -1)
                x2 = (x_i_0+x_j_0)/2
                y2 = (x_i_1+x_j_1)/2
                b = y2 - (m2*x2)
                
                idx += 1
                
                cnvs[idx, 2, 0] = b
                cnvs[idx, 2, 1] = m2
                cnvs[idx, 2, 2] = -1
        
        
        for k in range(1, n_cnvs+1):    
            
            for i in range(n):
                y_i = y[i]
                z_i = z[i]
                
                b = cnvs[k, 2, 0]
                m2 = cnvs[k, 2, 1]
                
                if (X[i,0] * m2) + b < X[i,1]:
                    #add to left
                    cnvs[k, 3, 0] += 1
                    cnvs[k, 4, 0] += y_i
                    if use_mse==1:
                        y_l[i] = y_i
                    else:
                        cnvs[k, 5, 0] += z_i
                else:
                    #add to right
                    cnvs[k, 6, 0] += 1
                    cnvs[k, 7, 0] += y_i
                    if use_mse==1:
                        y_r[i] = y_i
                    else:
                        cnvs[k, 8, 0] += z_i
            if use_mse==1:
                cnvs[k, 5, 0] = mse(y_l, mean(y_l))
                cnvs[k, 8, 0] = mse(y_r, mean(y_r))
                

            
            
            
    
    
                    
cdef void _sketch_gaussian(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        np.ndarray[DTYPE_t, ndim=2] xdim, 
        np.ndarray[DTYPE_t, ndim=3] cnvs,
        size_t i_start,
        size_t i_end,
        int use_mse):

    cdef size_t i, j, k, l
    cdef size_t n = X.shape[0]
    cdef size_t n_cnvs = <size_t> cnvs.shape[0]
    cdef double y_i, z_i
    cdef size_t idx = 0
    cdef double x_i_0, x_i_1, x_j_0, x_j_1, x_l_0, x_l_1
    cdef double halfpoint_0, halfpoint_1
    cdef double distance, distanceij
    cdef size_t id1, id2
    cdef double dist, dist1, dist2
    cdef double focal1_x, focal1_y, focal2_x, focal2_y
    cdef DTYPE_t [:] y_l = np.zeros(n)
    cdef DTYPE_t [:] y_r = np.zeros(n)

    with nogil:
        for i in range(n):
            x_i_0 = X[i,0]
            x_i_1 = X[i,1]
            
            
            for j in range(i+1, n):
                x_j_0 = X[j,0]
                x_j_1 = X[j,1]
                
                for l in range(n):
                    if l==i or l==j:
                        continue
                    idx +=1
                    x_l_0 = X[l,0]
                    x_l_1 = X[l,1]
                    halfpoint_0 = (x_i_0 + x_j_0) / 2.0
                    halfpoint_1 = (x_i_1 + x_j_1) / 2.0
                    distance = euclidean(halfpoint_0, halfpoint_1, x_l_0, x_l_1)
                    distanceij = euclidean(x_i_0, x_i_1, x_j_0, x_j_1)
                    if distance < distanceij:
                        continue
                    else:
                        cnvs[idx, 2, 0] = (i + i_start)
                        cnvs[idx, 2, 1] = (j + i_start)
                        cnvs[idx, 2, 2] = distance
        
        
        for k in range(1, n_cnvs+1):    
            
            for i in range(n):
                y_i = y[i]
                z_i = z[i]
                
                id1 = (<size_t>cnvs[k, 2, 0] - i_start)
                id2 = (<size_t>cnvs[k, 2, 1] - i_start)
                dist = cnvs[k, 2, 2]
                
                focal1_x = X[id1,0]
                focal1_y = X[id1,1]
                focal2_x = X[id2,0]
                focal2_y = X[id2,1]
                dist_1 = sqrt(square(X[i,0] - focal1_x) + square(X[i,1] - focal1_y))
                dist_2 = sqrt(square(X[i,0] - focal2_x) + square(X[i,1] - focal2_y))
                if (dist_1 + dist_2) < dist:
                    cnvs[k, 3, 0] += 1
                    cnvs[k, 4, 0] += y_i
                    if use_mse==1:
                        y_l[i] = y_i
                    else:
                        cnvs[k, 5, 0] += z_i
                else:
                    #add to right
                    cnvs[k, 6, 0] += 1
                    cnvs[k, 7, 0] += y_i
                    if use_mse==1:
                        y_r[i] = y_i
                    else:
                        cnvs[k, 8, 0] += z_i
            if use_mse==1:
                cnvs[k, 5, 0] = mse(y_l, mean(y_l))
                cnvs[k, 8, 0] = mse(y_r, mean(y_r))
    
    
    

def apply_tree(tree_ind, tree_val, X, y, output_type):
    if output_type == "index":
        return _apply_tree0(tree_ind, tree_val, X, y)
    else:
        return _apply_tree1(tree_ind, tree_val, X, y)

# output index
cdef np.ndarray[DTYPE_t, ndim=1] _apply_tree0(
                            np.ndarray[np.int_t, ndim=2] tree_ind, 
                            np.ndarray[DTYPE_t, ndim=2] tree_val, 
                            np.ndarray[DTYPE_t, ndim=2] X, 
                            np.ndarray[DTYPE_t, ndim=1] y):
    # Initialize node/row indicies
    cdef size_t i, t
    cdef size_t n_samples = X.shape[0]

    with nogil:
        for i in range(n_samples):
            t = 0
            while tree_ind[t,0] < 0:
                if isnan(X[i, tree_ind[t,1]]):
                    if tree_ind[t,2]==0:
                        t = tree_ind[t,3] 
                    else:
                        t = tree_ind[t,4] 
                else:
                    if X[i,tree_ind[t,1]] < tree_val[t,0]:
                        t = tree_ind[t,3]
                    else:
                        t = tree_ind[t,4]
            y[i] = tree_ind[t,5]
    return y

# output y values
cdef np.ndarray[DTYPE_t, ndim=1] _apply_tree1(
                            np.ndarray[np.int_t, ndim=2] tree_ind, 
                            np.ndarray[DTYPE_t, ndim=2] tree_val, 
                            np.ndarray[DTYPE_t, ndim=2] X, 
                            np.ndarray[DTYPE_t, ndim=1] y):
    #tree_ind: [isleaf, svar1, svar2, missing, left, right, index]
    #tree_val:  [sval1, sval2, sval3, sval4, sval5, out]
    # Initialize node/row indicies
    cdef size_t i, t
    cdef size_t n_samples = X.shape[0]
    cdef DTYPE_t dist, dist_1, dist_2
    cdef DTYPE_t focal1_x, focal1_y, focal2_x, focal2_y

    with nogil:
        for i in range(n_samples):
            t = 0
            while tree_ind[t,0] < 0:
                if isnan(X[i, tree_ind[t,1]]) | isnan(X[i, tree_ind[t,2]]):
                    if tree_ind[t,3]==0:
                        t = tree_ind[t,4] 
                    else:
                        t = tree_ind[t,5] 
                else:
                    if tree_ind[t, 2] == -1:
                        if X[i,tree_ind[t,1]] < tree_val[t,0]:
                            t = tree_ind[t,4]
                        else:
                            t = tree_ind[t,5]
                    elif tree_val[t, 2] == -1:
                        if tree_val[t,1] * X[i,tree_ind[t,1]] + tree_val[t,0] < X[i,tree_ind[t,2]]:
                            t = tree_ind[t,4]
                        else:
                            t = tree_ind[t,5]
                    else:
                        focal1_x = tree_val[t,0]
                        focal1_y = tree_val[t,1]
                        focal2_x = tree_val[t,2]
                        focal2_y = tree_val[t,3]
                        dist = tree_val[t,4]
                        dist_1 = sqrt(square(X[i,tree_ind[t,1]] - focal1_x) + square(X[i,tree_ind[t,2]] - focal1_y))
                        dist_2 = sqrt(square(X[i,tree_ind[t,1]] - focal2_x) + square(X[i,tree_ind[t,2]] - focal2_y))
                        if (dist_1 + dist_2) >= dist:
                            t = tree_ind[t,5]
                        else:
                            t = tree_ind[t,4]
            y[i] = tree_val[t,5]
    return y


