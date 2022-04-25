"""
This module defines the Bonsai class and its basic templates.
The Bonsai class implments splitting data and constructing decision rules.
User need to provide two additional functions to complete the Bonsai class:
- find_split()
- is_leaf()
"""
# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core._bonsaic import (
    reorder, 
    sketch,
    sketch_diagonal, 
    sketch_gaussian,
    apply_tree)
from bonsai.core._utils import (
    reconstruct_tree,
    get_xdim,
    get_cnvsn,
    get_cnvs,
    get_child_branch)
import numpy as np
import json
from joblib import parallel_backend, Parallel, delayed, cpu_count
from scipy.special import expit
import time
import logging

class Bonsai:

    def __init__(self, 
                find_split,
                is_leaf, 
                n_hist_max = 512, 
                subsample = 1.0, # subsample rate for rows (samples)
                random_state = None,
                z_type = "M2",
                n_jobs = -1,
                orthogonal = True,
                diagonal = True,
                gaussian = True,
                rm_outliers = True):

        self.find_split = find_split # user-defined
        self.is_leaf = is_leaf       # user-defined
        self.n_hist_max = n_hist_max
        self.subsample = np.clip(subsample, 0.0, 1.0)
        self.random_state = random_state
        self.z_type = z_type

        self.leaves = []
        self.feature_importances_ = None
        self.n_features_ = 0
        self.tree_ind   = np.zeros((1,6), dtype=np.int)
        self.tree_val   = np.zeros((1,2), dtype=np.float)
        self.mask       = None
        self.xdim       = None
        self.cnvs       = None
        self.cnvsn      = None
        self.n_jobs     = n_jobs
        if self.n_jobs < 0:
            self.n_jobs = cpu_count()
        self.orthogonal = orthogonal
        self.diagonal = diagonal
        self.gaussian = gaussian
        self.focalpoints = np.array([-1, -1, -1], dtype=np.float)

    def get_avc(self, X, y, z, i_start, i_end, parallel):
        n, m = X[i_start:i_end,:].shape
        y_i = y[i_start:i_end]
        z_i = z[i_start:i_end]
        self.cnvs[:,3:,:] = 0  # initialize canvas
        self.cnvsn[:,1:] = 0 # initialize canvas for NA
        use_mse = (self.z_type=="MSE")
        jj_start=0
        #print("fill canvas from "+str(i_start) +" to "+str(i_end))
        
        # TODO: smart guidance on "n_jobs"
        k = int(np.ceil(m/self.n_jobs))
        def psketch(i):
            j_start = i*k
            if j_start > m-1:
                return 1
            j_end = min(m, (i+1)*k)
            jj_start = int(self.xdim[j_start,4]*2)
            jj_end = int(self.xdim[j_end-1,4]*2 + 
                        self.xdim[j_end-1,3]*2)
            #print(jj_end)
            X_ij = X[i_start:i_end,j_start:j_end]
            xdim_j = self.xdim[j_start:j_end,:]
            cnvs_j = self.cnvs[jj_start:jj_end,:,:]
            cnvsn_j = self.cnvsn[j_start:j_end,:]
            sketch(X_ij, y_i, z_i, xdim_j, cnvs_j, cnvsn_j, use_mse)
            return 0

        #t0 = time.time()
        if self.orthogonal:
            parallel(delayed(psketch)(i) for i in range(self.n_jobs))
            jj_start = int(self.xdim[m-1,4]*2 + 
                        self.xdim[m-1,3]*2)
        #t1 = time.time() - t0
        #print(i_end-i_start, t1)

            
        #print(jj_start)
            # start after offset + n_bin*2 of last feature
        jj_end = jj_start
        X_ij = X[i_start:i_end,:2]
        #print(X_ij.shape[0])
        xdim_j = self.xdim[:2,:]
        if self.diagonal:
            jj_end += n*(n-1)
            cnvs_j = self.cnvs[jj_start:jj_end,:,:]
            sketch_diagonal(X_ij, y_i, z_i, xdim_j, cnvs_j, use_mse)
            #print(jj_end)
            #print(self.cnvs[(jj_start-3):(jj_start+4),:,:])
            #print(self.cnvs[(jj_end-2):(jj_end+2),:,:])
        if self.gaussian:
            jj_start = jj_end
            jj_end += n*(n-1)*(n-2)
            cnvs_j = self.cnvs[jj_start:jj_end,:,:]
            #print("sketch gaussian from "+str(i_start) + " to "+str(i_end))
            sketch_gaussian(X_ij, y_i, z_i, xdim_j, cnvs_j, i_start, i_end, use_mse)
            #print(jj_end)
            #print(self.cnvs[jj_start:(jj_start+4),:,:])
            #print(self.cnvs[(jj_end-2):(jj_end+2),:,:])
            
        
        return self.cnvs

    def split_branch(self, X, y, z, branch, parallel):
        """Splits the data (X, y) into two children based on 
           the selected splitting variable and value pair.
        """

        i_start = branch["i_start"]
        i_end = branch["i_end"]
   
        # Get AVC-GROUP
        avc = self.get_avc(X, y, z, i_start, i_end, parallel)
        #print(avc.shape)
        if avc.shape[0] < 2:
            branch["is_leaf"] = True
            return [branch]
        
        y_i = y[i_start:i_end]
        parent_mse=0
        if self.z_type == "MSE":
            parent_mse = np.sum((y_i - np.mean(y_i))**2) / len(y_i)
        # Find a split SS: selected split
        ss = self.find_split(avc, X, y, z, i_start, i_end, parent_mse)     
        if (not isinstance(ss, dict) or
            "selected" not in ss):
            branch["is_leaf"] = True
            return [branch]

        svar = np.array(ss["selected"][1], dtype=np.int32)
        sval = ss["selected"][2]
        missing = ss["selected"][9]
        
        #print(svar)
        #print(sval)
        if (svar.shape == 0) | (sval.shape ==0):
            print(svar, sval)

        if (svar[1]!=-1) & (sval[2] != -1):
            self.focalpoints = np.vstack((self.focalpoints,[int(sval[0]), X[int(sval[0]), 0], X[int(sval[0]), 1]]))
            self.focalpoints = np.vstack((self.focalpoints,[int(sval[1]), X[int(sval[1]), 0], X[int(sval[1]), 1]]))
            #print("finding focal points:")
            #print(self.focalpoints[-2:])
        
        i_split = reorder(X, y, z, i_start, i_end, 
                            svar, sval, np.array(missing, dtype=np.int32))

        if i_split==i_start or i_split==i_end:
            # NOTE: this condition may rarely happen
            #       We just ignore this case, and stop the tree growth
            branch["is_leaf"] = True
            print("reorder gave i_split: " + str(i_split))
            return [branch]
            
        
        left_branch = get_child_branch(ss, branch, i_split, "@l")
        left_branch["is_leaf"] = self.is_leaf(left_branch, branch)

        right_branch = get_child_branch(ss, branch, i_split, "@r")
        right_branch["is_leaf"] = self.is_leaf(right_branch, branch) 
        return [left_branch, right_branch]

    def grow_tree(self, X, y, z, branches, parallel):
        """Grows a tree by recursively partitioning the data (X, y)."""
        branches_new = []
        leaves_new = []
        print('.. grow tree...')
        for i, branch in enumerate(branches):
            for child in self.split_branch(X, y, z, branch, parallel):
                if child["is_leaf"]:
                    leaves_new.append(child)
                else:
                    branches_new.append(child)
        return branches_new, leaves_new

    def fit(self, X, y, init_cnvs=True, parallel=None): 
        """Fit a tree to the data (X, y)."""
        print("Fitting...")
        n, m = X.shape
        X = X.astype(np.float, order="C", copy=True)
        y = y.astype(np.float, order="C", copy=True)
        if self.z_type=="M2":
            z = np.square(y)
        elif self.z_type=="Hessian": # bernoulli hessian
            p = expit(y) 
            z = p * (1.0 - p)
        elif self.z_type=="MSE":
            z = np.array([np.sum((y-np.mean(y))**2) / y.shape[0] for i in range(n)])
        else:
            z = np.ones(n) 
 
        if self.subsample < 1.0:
            np.random.seed(self.random_state)
            self.mask = (np.random.rand(n) < self.subsample)
            print('mask: ')
            print(self.mask)
            X = X[self.mask,:]
            y = y[self.mask]
            z = z[self.mask]
            n, m = X.shape
        else:
            self.mask = np.full(n, True, dtype=np.bool)

        self.n_features_ = m

        branches = [{"_id": "ROOT",
                    "is_leaf": False,
                    "depth": 0,
                    "eqs": [],
                    "i_start": 0,
                    "i_end": n,
                    "y": np.mean(y),
                    "y_lst": [], 
                    "n_samples": n}]

        if init_cnvs:
            self.init_cnvs(X)

        self.leaves = []
        if self.xdim is None or self.cnvs is None or self.cnvsn is None:
            logging.error("canvas is not initialized. no tree trained")
            return 1

        if parallel is None:
            with Parallel(n_jobs=self.n_jobs, prefer="threads") as parallel:
                while len(branches) > 0:
                    branches, leaves_new = self.grow_tree(X, y, z, 
                                                branches, parallel)
                    self.leaves += leaves_new
        else:
            # parallel-context is already given
            while len(branches) > 0:
                branches, leaves_new = self.grow_tree(X, y, z, 
                                            branches, parallel)
                self.leaves += leaves_new
        #print("fitted leaves: ")
        #print(self.leaves)
        # integer index for leaves (from 0 to len(leaves))
        for i, leaf in enumerate(self.leaves): 
            leaf["index"] = i 
        #self.update_feature_importances()
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves, self.focalpoints)
        self.print_splits(self.tree_ind, self.tree_val, y)
        
        return 0
    
    def print_splits(self, tree_ind, tree_val, y):
        for t in range(len(tree_ind)):
            
            if tree_ind[t,0]==-1:
                parent = t-1
                parent_measure = tree_val[(t-1),6]
                if parent ==-1:
                    parent_measure = (np.sum(y**2)/y.shape[0]) - (np.mean(y)**2)
                gain = tree_val[t,6]
                if tree_ind[t,2]==-1:
                    print(str(gain)+ " X["+str(tree_ind[t,1])+"] >= "+str(tree_val[t,0]))
                elif tree_val[t,2]==-1:
                    print(str(gain)+ " X["+str(tree_ind[t,2])+"] >= X["+ str(tree_ind[t,1])+"] * "+str(tree_val[t,1]) + " + "+str(tree_val[t,0]))
                else:
                    #d(X[0], [0.20560896 0.22449331]) + d(X[1], [0.2980301  0.79909647]) >= 0.6355280056423498
                    print(str(gain)+ " d(X["+str(tree_ind[t,1])+"], ["+str(tree_val[t,0]) + " " +str(tree_val[t,1])+"])" +
                          " + d(X["+str(tree_ind[t,2])+"], ["+str(tree_val[t,2]) + " " +str(tree_val[t,3])+"]) >= " + str(tree_val[t,4]))

                          
    def predict(self, X_new, output_type="response"):
        """Predict y by applying the trained tree to X."""
        X_new = X_new.astype(np.float)
        n, m = X_new.shape
        y = np.zeros(n, dtype=np.float)
        print('predict') 
        #print(self.tree_ind, self.tree_val)
        out = apply_tree(self.tree_ind, self.tree_val, X_new, y, output_type)
        return out 

    def init_cnvs(self, X):
        self.xdim = get_xdim(X, self.n_hist_max, self.rm_outliers)
        self.cnvs = get_cnvs(self.xdim, self.orthogonal, self.diagonal, self.gaussian)
        self.cnvsn = get_cnvsn(self.xdim)

    def set_cnvs(self, xdim, cnvs, cnvsn):
        self.xdim = xdim
        self.cnvs = cnvs
        self.cnvsn = cnvsn
        self.cnvs[:,3:,:] = 0  # initialize canvas
        self.cnvsn[:,1:] = 0 # initialize canvas for NA
 
    def get_cnvs(self):
        return self.xdim, self.cnvs, self.cnvsn

    def is_stochastic(self):
        return self.subsample < 1.0

    def get_mask(self):
        return self.mask
    
    def get_oob_mask(self):
        """Returns a mask array for OOB samples"""
        return ~self.mask

    def get_ttab(self):
        """Returns tree tables (ttab). 
            ttab consists of tree_ind (np_array) and tree_val (np_array).
            tree_ind stores tree indices - integer array. 
            tree_val stores node values - float array.
        """
        return self.tree_ind, self.tree_val

    def dump(self, columns=[], compact=False):
        """Dumps the trained tree in the form of array of leaves"""
        def default(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        n_col = len(columns)
        for leaf in self.leaves:
            for eq in leaf["eqs"]:
                if isinstance(eq["svar"], list):
                    if (eq["svar"][0] < n_col) & (eq["svar"][1] < n_col):
                        eq["name"] = (columns[int(eq["svar"][0])], columns[int(eq["svar"][1])])
                else:
                    if eq["svar"] < n_col:
                        eq["name"] = columns[int(eq["svar"])]
        out = json.loads(json.dumps(self.leaves, default=default))
        if compact:
            suplst = ["i_start", "i_end", "depth",
                        "_id", "n_samples", "y_lst", 
                        "is_leaf", "prune_status"] # suppress
            out_cmpct = []
            for leaf in out:
                for key in suplst:
                    leaf.pop(key, None)
                out_cmpct.append(leaf) 
            return out_cmpct
        else:
            return out

    def load(self, leaves, columns=None):
        """Loads a new tree in the form of array of leaves"""
        self.leaves = leaves
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves, self.focalpoints)
        return None

    def get_sibling_id(self, leaf_id):
        """Returns a sibling ID for the given leaf_id.
           Siblings are the nodes that are at the same level 
            with the same parent node.
        """
        sibling_id = None
        if leaf_id[-1]=="L":
            sibling_id = leaf_id[:-1] + "R"
        elif leaf_id[-1]=="R":
            sibling_id = leaf_id[:-1] + "L"
        sibling_leaf = [leaf for leaf in self.leaves 
                        if leaf["_id"]==sibling_id]
        if len(sibling_leaf) == 0:
            sibling_id = None
        return sibling_id 

    def get_sibling_pairs(self):
        """Returns an array of sibling pairs. 
            For more info, see the get_sibling_id
        """
        id2index = {leaf["_id"]:i for i, leaf in enumerate(self.leaves)}
        leaf_ids = [k for k in id2index.keys()]
        sibling_pairs = []
        while len(leaf_ids) > 0:
            leaf_id = leaf_ids.pop()
            sibling_id = self.get_sibling_id(leaf_id) 
            if sibling_id is not None:
                if sibling_id in leaf_ids:
                    leaf_ids.remove(sibling_id)
                sibling_pairs.append((id2index[leaf_id], 
                                      id2index[sibling_id]))
            else:
                sibling_pairs.append((id2index[leaf_id], None))
        return sibling_pairs
   
    def get_feature_importances(self):
        return self.feature_importances_
 
    def update_feature_importances(self):
        """Returns a modified feature importance.
            This formula takes into account of node coverage and leaf value.
            NOTE: This is different from regular feature importances that
                are used in RandomForests or GBM.
            For more info, please see the PaloBoost paper.
        """
        if self.n_features_ == 0:
            return None
        self.feature_importances_ = np.zeros(self.n_features_)
        cov = 0
        J = len(self.leaves)
        if J > 0:
            for j, leaf in enumerate(self.leaves):
                gamma_j = np.abs(leaf["y"])
                cov_j = leaf["n_samples"]
                cov += cov_j
                eff_j = cov_j*gamma_j
                for eq in leaf["eqs"]:
                    self.feature_importances_[int(eq["svar"])] += eff_j
            self.feature_importances_ /= J
            self.feature_importances_ /= cov
        return self.feature_importances_ 

