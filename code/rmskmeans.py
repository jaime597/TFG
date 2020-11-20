#!/usr/bin/env python3
# coding: utf-8

# MIT License
# 
# Copyright (c) 2017 Behrouz Babaki
# Copyright (c) 2020 Jaime Meléndez / Carlos Tellería
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pulp
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

""" return distance^2 between points"""
def l2_distance(point1, point2):
    return sum((float(i)-float(j))**2 for (i,j) in zip(point1, point2))


""" returns best cluster for a point (nearest center)"""
def get_best_cluster(point, centers):
    distances = map(lambda x: l2_distance(point,x), centers)
    df = pd.DataFrame(distances)
    return df.idxmin(axis=0).values[0]


class subproblem(object):
    def __init__(self, centroids, data, min_size):

        self.centroids = centroids
        self.data = data[['x','y']].values.tolist()
        self.min_size = min_size
        self.n = len(self.data)
        self.k = len(centroids)
        self.create_model()

    def create_model(self):
        def distances(assignment):
            return l2_distance(self.data[assignment[0]], self.centroids[assignment[1]])

        clusters = list(range(self.k))
        assignments = [(i, j)for i in range(self.n) for j in range(self.k)]

        # outflow variables for data nodes
        self.y = pulp.LpVariable.dicts('data-to-cluster_assignments',
                                  assignments,
                                  lowBound=0,
                                  upBound=1,
                                  cat=pulp.LpInteger)

        # outflow variables for cluster nodes
        self.b = pulp.LpVariable.dicts('cluster_outflows',
                                  clusters,
                                  lowBound=0,
                                  upBound=self.n-self.min_size,
                                  cat=pulp.LpContinuous)

        # create the model
        self.model = pulp.LpProblem("Model_for_assignment_subproblem", pulp.LpMinimize)

        # objective function
        self.model += pulp.lpSum(distances(assignment) * self.y[assignment] for assignment in assignments)

        # flow balance constraints for data nodes
        for i in range(self.n):
            self.model += pulp.lpSum(self.y[(i, j)] for j in range(self.k)) == 1

        # flow balance constraints for cluster nodes
        for j in range(self.k):
            self.model += pulp.lpSum(self.y[(i, j)] for i in range(self.n)) - self.min_size == self.b[j]

        # flow balance constraint for the sink node
        self.model += pulp.lpSum(self.b[j] for j in range(self.k)) == self.n - (self.k * self.min_size)


    def solve(self):
        self.status = self.model.solve()

        clusters = None
        if self.status == 1:
            clusters= [-1 for i in range(self.n)]
            for i in range(self.n):
                for j in range(self.k):
                    if self.y[(i, j)].value() > 0:
                        clusters[i] = j
        return clusters

def initialize_centers(dataset, k):
    ids = list(range(len(dataset)))
    random.shuffle(ids)
    ''' tomamos como centros de los clusters las k primeras coordenadas del dataset'''
    return (dataset.iloc[ids[:k]][['x','y']]).values.tolist()

def compute_centers(dataset): #dataset must be['x','y','cluster']
    dt2 = dataset.groupby(['cluster']).agg({'cluster':'count', 'x': 'sum', 'y': 'sum'}).rename(columns={'cluster':'counter'})
    dt2['x'] = dt2.apply(lambda row: row.x/row.counter, axis=1)
    dt2['y'] = dt2.apply(lambda row: row.y/row.counter, axis=1)
    return dt2[['x','y']].values.tolist()

def minsize_kmeans(dataset, k, min_size=0):

    centers = initialize_centers(dataset, k)
    clusters = [-1] * len(dataset) 
    converged = False
    
    while not converged:
        m = subproblem(centers, dataset, min_size)
        clusters_ = m.solve()
        if not clusters_:
            return None, None
        
        dataset['cluster'] = clusters_
        centers = compute_centers(dataset[['x','y','cluster']])
        converged = True
        i = 0
        while converged and i < len(dataset):
            if clusters[i] != clusters_[i]:
                converged = False
            i += 1
        clusters = clusters_
    
    return clusters, centers



def read_data(datafile):
    data = []
    with open(datafile, 'r') as f:
        for line in f:
            line = line.strip()
            if line != '':
                d = [float(i) for i in line.split()]
                data.append(d)
    return data


""" cluster_quality = mean cuadratic distance between cluster centers """
def cluster_quality(cluster):
    if len(cluster) == 0:
        return 0.0

    quality = 0.0
    for i in range(len(cluster)):
        for j in range(i, len(cluster)):
            quality += l2_distance(cluster[i], cluster[j])
    return quality / len(cluster)



def compute_quality(data, cluster_indices):
    clusters = dict()
    for i, c in enumerate(cluster_indices):
        if c in clusters:
            clusters[c].append(data.iloc[i][['x','y']])
        else:
            clusters[c] = [data.iloc[i][['x','y']].values]
    return sum(cluster_quality(c) for c in clusters.values())


"""
data must be a DataFrame with columns (id, x, y)
k = number of clusters to find
iter = num of iterations. Default = 1
fraction = fraction of data records to use in cluster finding. Default=0.1 (10%)
images = Generate or not plot from results
name = Name of execution, for output files 
"""
def rmskmeans(data, k, min_size=1, n_iter=1, fraction=0.1, images=False, name='name') :

    filtered_data = data[['id','x','y']].dropna()
    
    idata = list(range(len(filtered_data)))
    random.shuffle(idata)

    # sample selection
    selected = filtered_data.sample(frac=fraction)
    selected.reset_index(drop=True)
    print('sample size: ', len(selected) )
    
    best = None
    best_clusters = None
    for i in range(n_iter):
        print("computing minsize_kmeans iteracion: %s"%(i))
        clusters, centers = minsize_kmeans(selected, k, min_size*fraction)

        if clusters:
            print("computing solution quality")
            quality = compute_quality(selected, clusters)
            if not best or (quality < best):
                best = quality
                best_clusters = clusters

    if best:
        selected['cluster'] = best_clusters
        pd.DataFrame(centers, columns=['x','y']).to_csv('%s_centers.csv'%(name))

        if images:
            plt.figure(0)
            plt.scatter(selected['x'], selected['y'], c=selected['cluster'], cmap='tab20b')
            plt.savefig('%s_select.png'%(name))

        data['cluster'] = data.apply(lambda row: get_best_cluster([row.x, row.y], centers), axis=1)    
        data.to_csv('%s_rkm.csv'%(name))

        if images:
            plt.figure(1)
            plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='tab20b')
            plt.savefig('%s_rkm.png'%(name))

        distribution = data[['id','cluster']].groupby('cluster').count().sort_values(by=['id']).reset_index(drop=True).rename(columns={'id': "size"})
        distribution.to_csv('%s_distrib.csv'%(name))
        if images:
            #fig = px.bar(distribution, x='cluster', y='size')
            #fig.show()
            plt.figure(2)
            plt.bar(distribution.index,distribution['size'])
            plt.savefig('%s_dist.png'%(name))
    else:
        print('no clustering found')


       
        
if __name__ == '__main__':
    
    tic = time.process_time()
    tstic = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', help='file containing the coordinates of instances')
    parser.add_argument('k', help='number of clusters', type=int)
    parser.add_argument('min_size', help='minimum size of each cluster', type=int)  
    parser.add_argument('-f', '--FRACTION', help='fraction of data to find clusters', type=float)  
    parser.add_argument('-n', '--NUM_ITER', type=int,
                        help='run the algorithm for NUM_ITER times and return the best clustering',
                        default=1)
    parser.add_argument('-o', '--OUTFILE', help='store the result in OUTFILE',
                        default='')
    args = parser.parse_args()    
    
    print("reading dataset")
    data = pd.read_csv(args.datafile, sep='|')
    rmskmeans(data, args.k, min_size=args.min_size, n_iter=args.NUM_ITER, fraction=args.FRACTION, images=True, name=args.OUTFILE)
        
    toc = time.process_time()
    tstoc = time.time()
    
    with open('%s.met'%(args.OUTFILE), 'w') as f:
        print('execution name: ', args.OUTFILE, end='\n', file=f)
        print('num. clusters: ', args.k, end='\n', file=f)
        print('min cluster size: ', args.min_size, end='\n', file=f)
        print('dataset size : ', len(data), end='\n', file=f)
        print('sample fraction : ', args.FRACTION, end='\n', file=f)
        print('# iterations: ', args.NUM_ITER, end='\n', file=f)
        print('process_time: %f'%(toc - tic), end='\n', file=f)     
        print('execution_time: %f'%(tstoc - tstic), end='\n', file=f)     

    print('process_time: %f'%(toc - tic))     
    f.close()   