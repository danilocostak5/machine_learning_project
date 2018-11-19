import pandas as pd
import numpy as np
import math as m
import pickle
import warnings

#ignore warnings
warnings.filterwarnings('ignore')

class ClusterMaker:

    
    #n is the number of elements to be clustered
    #k is the number of clusters
    #p is the number of views
    #q is the number of elements of a cluster prototype
    def __init__(self, filename, k, q, m, s, max_iter, read_files=False):
        self.k = k
        self.q = q
        self.p = 2
        self.m = m
        self.s = s
        self.max_iter = max_iter
        self.epsilon = 0.0000000001
        
        self.cost_evolution = []
        
        if read_files:
            print 'Reading input file'
            self.read_from_csv(filename)
            ClusterMaker.print_data_overview(self.raw_data)

            #Create the separated views
            self.create_views()

            #Create the dissimilarity matrix for both views
            self.diss_matrix_1 = ClusterMaker.calculate_diss_matrix(self.view_1)
            self.diss_matrix_2 = ClusterMaker.calculate_diss_matrix(self.view_2)

            print 'Saving dissimilarity matrices'

            out_1 = open('diss_matrix_1.pickle', 'wb')
            out_2 = open('diss_matrix_2.pickle', 'wb')
            pickle.dump(self.diss_matrix_1, out_1)
            pickle.dump(self.diss_matrix_2, out_2)
            out_1.close()
            out_2.close()

            print 'Matrices saved'
            
        else:
            print 'Loading previous matrices'
            self.diss_matrix_1 = pickle.load(open('diss_matrix_1.pickle','rb'))
            self.diss_matrix_2 = pickle.load(open('diss_matrix_2.pickle','rb'))

        self.diss = [self.diss_matrix_1, self.diss_matrix_2]
        
        print 'Dissimilarity matrices calculated'
        self.n = np.shape(self.diss_matrix_1)[0]
        
    def run_clustering(self):
        self.initialize_clustering()
        self.cost_evolution = []
        
        for t in xrange(self.max_iter):
            print 'Iteration', t, '---------'
            print 'Updating U'
            self.update_U()
            cost = self.cost()
            self.cost_evolution.append(cost)
            print 'New Cost', cost
            print 'Updating G'
            self.update_G()
            print 'Updating Lambda'
            self.update_Lambda()
            if t > 0:
                print 'Cost evolution', self.cost_evolution[-2] - cost 
                if self.cost_evolution[-2] - cost < self.epsilon:
                    print 'Cost is stationary'
                    break
            
    def read_from_csv(self, filename):
        rd = pd.read_csv(filename, sep=",", header=2)
        rd = rd.values #Numpy array
        self.raw_data = rd

    def create_views(self):
        size_view_1 = 9
        size_view_2 = 10
        num_rows = np.shape(self.raw_data)[0]
        self.view_1 = self.raw_data[np.ix_(range(num_rows),range(size_view_1))]
        self.view_2 = self.raw_data[np.ix_(range(num_rows),range(size_view_1, size_view_1 + size_view_2))]
        print 'View 1 shape', np.shape(self.view_1)
        print 'View 2 shape', np.shape(self.view_2)
        
    def initialize_clustering(self):
        n = self.n
        k = self.k
        p = self.p
        q = self.q
        
        self.U = np.ones((n, k))
        self.Lambda = np.ones((p, k))

        #shuffle indexes to randomly initialize cluster prototypes
        all_indexes = range(n)
        np.random.shuffle(all_indexes)

        self.G = np.zeros((k, q))

        for i in xrange(self.k):
            self.G[i] = np.array(all_indexes[i*q:(i+1)*q])
        

    def dist_to_cluster(self, elem_index, cluster_index):
        summ = 0
        for p in xrange(self.p):
            partial_sum = 0
            for j in xrange(self.q):
                cluster = self.G[cluster_index]
                partial_sum += self.diss[p][elem_index, cluster[j]]
            summ += self.Lambda[p][cluster_index]*partial_sum
        return summ

    def cost_by_cluster(self, cluster_index):
        summ = 0
        for i in xrange(self.n):
                summ += pow(self.U[i][cluster_index],self.m)*self.dist_to_cluster(i, cluster_index)
        return summ
    
    def cost(self):
        summ = 0
        for k in xrange(self.k):
            summ += self.cost_by_cluster(k)
        return summ
    
    def update_U(self):
        pot_term = 1.0/(self.m - 1.0)
        for k in xrange(self.k):
            for i in xrange(self.n):
                summ = 0
                for h in xrange(self.k):
                    summ += pow(self.dist_to_cluster(i, k)/self.dist_to_cluster(i, h), pot_term) 
                
                self.U[i][k] = pow(summ, -1)

    def part_lambda_dist(self, view_index, cluster_index):
        summ = 0
        for i in xrange(self.n):
            partial_sum = 0
            for j in xrange(self.q):
                cluster = self.G[cluster_index]
                partial_sum += self.diss[view_index][i, cluster[j]]
            summ += pow(self.U[i][cluster_index], self.m)*partial_sum
        return summ
    
    def update_Lambda(self):
        for i in xrange(self.p):
            for k in xrange(self.k):
                prod = 1
                for j in xrange(self.p):
                    prod *= self.part_lambda_dist(j, k)
                prod = pow(prod, 1.0/float(self.p))
                self.Lambda[i][k] = prod/self.part_lambda_dist(i, k)

    def part_G_sum(self, cluster_index):
        summ = 0
        n_rows = range(self.n)
        mat_U = np.transpose(pow(self.U, self.m))
        mat_diss = 0
        for p in xrange(self.p):
            mat_diss += self.Lambda[p][cluster_index]*self.diss[p]
            
        dot_val = np.dot(mat_U, mat_diss)
        
        return dot_val[cluster_index].tolist()

    @staticmethod
    def pick_q(list_var, forbidden, q):
        num_picked = 0
        picked = []
        new_forbidden = []
        
        for l in list_var:
            if not l[0] in forbidden:
                picked.append(l[0])
                num_picked += 1
            if num_picked == q:
                break
        return picked
    
    def update_G(self):
        all_indexes = range(self.n)
        all_index_dist = [[i, 0] for i in all_indexes]
        forbidden = []
        for k in xrange(self.k):
            len_id = len(all_index_dist)
            all_distances = self.part_G_sum(k)
            all_index_dist = [[i, d] for (i,d) in enumerate(all_distances)]
            all_index_dist = sorted(all_index_dist, key= lambda x : x[1])
            picked = ClusterMaker.pick_q(all_index_dist, forbidden, self.q)
            forbidden = forbidden + picked
            self.G[k] = np.array(picked)
            
    @staticmethod
    def soft_to_hard_cluster(U):
        n_rows = np.shape(U)[0]
        for i in xrange(n_rows):
            max_i = np.argmax(U[i])
            U[i] = 0*U[i]
            U[i][max_i] = 1
        return U

    @staticmethod
    def calculate_cont_matrix(U1, U2):
        U1 = np.transpose(U1)
        U2 = np.transpose(U2)
        
        k = np.shape(U1)[0]
        
        cont_matrix = np.zeros((k, k))
        for i in xrange(k):
            for j in xrange(k):
                l1 = U1[i]
                l2 = U2[j]
                cont_matrix[i][j] = np.dot(l1, l2)
        return cont_matrix

    @staticmethod
    def comb2(n):
        return n*(n-1)/2.0

    def adjusted_rand_index(self, other_U):
        n = self.n
        hard_U = ClusterMaker.soft_to_hard_cluster(self.U)
        cont_matrix = ClusterMaker.calculate_cont_matrix(hard_U, other_U)
        a = np.sum(cont_matrix, axis=1)
        b = np.sum(cont_matrix, axis=0)
        sum_combs_a = np.sum([ClusterMaker.comb2(i) for i in a])
        sum_combs_b = np.sum([ClusterMaker.comb2(i) for i in b])
        sum_comb2_cont = np.sum(ClusterMaker.comb2(cont_matrix))
        comb_n = ClusterMaker.comb2(n)
        
        rand_ind = (sum_comb2_cont - (sum_combs_a*sum_combs_b)/comb_n)
        rand_ind /= (0.5*(sum_combs_a + sum_combs_b) - (sum_combs_a*sum_combs_b)/comb_n)
        return rand_ind
    
    @staticmethod
    def cvt_np_array(matrix):
        if type(matrix) != np.ndarray:
            matrix = np.array(matrix)
        return matrix

    @staticmethod
    def print_data_overview(raw_data):    
        raw_data = ClusterMaker.cvt_np_array(raw_data)
        
        num_elems = np.shape(raw_data)[0]
        num_vars = np.shape(raw_data)[1]

        print 'Num of elements:', num_elems
        print 'Num of vars:', num_vars

    @staticmethod
    def euclid_dist(vect1, vect2):
        if len(vect1) != len(vect2):
            raise ValueError('The size of the vectors must be equal')

        vect1 = ClusterMaker.cvt_np_array(vect1)
        vect2 = ClusterMaker.cvt_np_array(vect2)
        
        diff = vect1 - vect2
        
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def calculate_diss_matrix(raw_data):
        raw_data = ClusterMaker.cvt_np_array(raw_data)
        
        num_elems = np.shape(raw_data)[0]
        diss_matrix = np.zeros((num_elems, num_elems))

        for i in xrange(num_elems):
            for j in xrange(num_elems):
                elem1 = raw_data[i]
                elem2 = raw_data[j]
            
                diss_matrix[i][j] = ClusterMaker.euclid_dist(elem1, elem2)    

        return diss_matrix

    def save_matrices(self):
        hard_cluster = ClusterMaker.soft_to_hard_cluster(self.U)
        other_U = pickle.load(open('apriori_U.pickle', 'rb'))
        rand = self.adjusted_rand_index(other_U)
        return [self.U, self.Lambda, self.G, hard_cluster, rand]
