def database(name, m, sub_graph, edges_, adj_matrix, num_cl, g_type, F):
    print(f'Processing set {m}')

    file = open(name + '_database_'+str(m)+'.csv', 'w')
    file.write('dataset' + "\t" + 'u' + "\t" + 'v' + "\t" 
            + 'short path length'+ "\t" + 'nr of 2paths'+ "\t" + 'nr of 3paths' +  "\t"
            + 'Jaccard'+ "\t" + 'Salton'+ "\t" + 'Sorensen' + "\t" + '3-Jaccard'+ "\t" + '3-Salton'+ "\t" + '3-Sorensen' +  "\t" 
            + 'hub depressed'+ "\t" + 'hub promoted' + "\t"
            + 'cos'+ "\t" + 'l1'+ "\t" + 'l2' + "\t"
            + 'corr'+ "\t" 
            + "field0" + "\t" + "field1" + "\t" + "field2" #+ "\t" + "field3" + "\t" + "field4" + "\t" + "field5" #+ "\t" + "field6"
            #+ "\t"  + "field6" + "\t" + "field7" + "\t" + "field8" + "\t" + "field9" + "\t" + "field10" + "\t" + "field11"
             #  + "\t"  + "field12" + "\t" + "field13" + "\t" + "field14" + "\t" + "field15" + "\t" + "field16" + "\t" + "field17"
               + "\t" + "Adamic Andar"  +  "\t" + "label" +"\n")
    

    for edge in edges_:

        u, v = edge
            
        if sub_graph.has_edge(u,v):
            sub_graph.remove_edge(u, v)
            try:
                path = nx.shortest_path(sub_graph, u, v)
            except:
                path = [0]
            sub_graph.add_edge(u, v) 
        else:
            try:
                path = nx.shortest_path(sub_graph, u, v)
            except:
                path = [0]

        paths2, paths3 = count_simple_paths_for_edge_sparse(adj_matrix, u, v)
   
        try:
            F1 = jaccard_index(sub_graph,u,v)
        except:
            F1 = 0

        try:
            F2 = salton_index(sub_graph,u,v)
        except:
            F2 = 0

        try:
            F3 = sorensen_index(sub_graph,u,v)
        except:
            F3 = 0

        try:   
            F4 = float(jaccard_3_paths(sub_graph, u, v, paths3))
        except:
            F4 = 0

        try:    
            F5 = float(salton_3_paths(sub_graph, u, v, paths3))
        except:
            F5 = 0

        try:
            F6 = float(sorensen_3_paths(sub_graph, u, v, paths3))
        except:
            F6 = 0   

        try:
            A = list(nx.adamic_adar_index(sub_graph, [(u, v)]))[0][2]
        except:
            A = 0
                
        Cl = create_array(Y[u], Y[v], num_cl)
        
        hdi = hub_depressed_index(sub_graph, u, v)
        hpi = hub_promoted_index(sub_graph, u, v)

        # Distance & Similarity
        f_u = F[u]
        f_v = F[v]
        cosine_sim = safe_cosine_similarity(f_u, f_v)
        l2_dist = euclidean(f_u, f_v)
        l1_dist = cityblock(f_u, f_v)
        
        if np.std(f_u) == 0 or np.std(f_v) == 0:
            pearson_corr = 0.0  # or np.nan if you want to flag it
        else:
            pearson_corr = np.corrcoef(f_u, f_v)[0, 1]

    

        if m in [-1,-2,-3]:
            label = 1
        else:
            label = 0


        file.write(name + "\t" + str(u) + "\t" + str(v) + "\t" 
                   + str(len(path)-1) + "\t" + str(paths2) + "\t" + str(paths3) + "\t" 
                   + str(F1) + "\t" + str(F2) + "\t" + str(F3) + "\t" 
                   + str(F4) + "\t" + str(F5) + "\t" + str(F6) + "\t" 
                   + str(hdi) + "\t" + str(hpi) + "\t" 
                   + str(cosine_sim) + "\t" + str(l1_dist) + "\t" + str(l2_dist) + "\t"
                   + str(pearson_corr) + "\t" 
                   + str(Cl[0]) + "\t" + str(Cl[1]) + "\t" + str(Cl[2]) #+ "\t" + str(Cl[3]) + "\t" + str(Cl[4]) + "\t" + str(Cl[5])# + "\t" + str(Cl[6])
               # + "\t" + str(Cl[6]) + "\t" + str(Cl[7]) + "\t" + str(Cl[8]) + "\t" + str(Cl[9]) + "\t" + str(Cl[10]) + "\t" + str(Cl[11])
                  # + "\t" + str(Cl[12]) + "\t" + str(Cl[13]) + "\t" + str(Cl[14]) + "\t" + str(Cl[15]) + "\t" + str(Cl[16]) + "\t" + str(Cl[17])
                   + "\t" + str(A) + "\t" + str(label) + "\n")
        
        if g_type == 'undirected':
            file.write(name + "\t" + str(v) + "\t" + str(u) + "\t" 
                   + str(len(path)-1) + "\t" + str(paths2) + "\t" + str(paths3) + "\t" 
                   + str(F1) + "\t" + str(F2) + "\t" + str(F3) + "\t" 
                   + str(F4) + "\t" + str(F5) + "\t" + str(F6) + "\t" 
                   + str(hdi) + "\t" + str(hpi) + "\t" 
                   + str(cosine_sim) + "\t" + str(l1_dist) + "\t" + str(l2_dist) + "\t"
                   + str(pearson_corr) + "\t"    
                   + str(Cl[0]) + "\t" + str(Cl[1]) + "\t" + str(Cl[2])# + "\t" + str(Cl[3]) + "\t" + str(Cl[4]) + "\t" + str(Cl[5]) #+ "\t" + str(Cl[6])
                #+ "\t" + str(Cl[6]) + "\t" + str(Cl[7]) + "\t" + str(Cl[8]) + "\t" + str(Cl[9]) + "\t" + str(Cl[10]) + "\t" + str(Cl[11])
                 #  + "\t" + str(Cl[12]) + "\t" + str(Cl[13]) + "\t" + str(Cl[14]) + "\t" + str(Cl[15]) + "\t" + str(Cl[16]) + "\t" + str(Cl[17])                   
                       + "\t" + str(A) + "\t" + str(label) + "\n")
            

    file.close()
    print('done')  
    return sub_graph

import torch
import networkx as nx
import random
import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import rankdata, spearmanr, skew
from dataset_processing import *
from functions import *


init_time = time.time()
name = 'Pubmed'
edge_list_init, node_list, Y, g_type, F = dataset_import(name)
if g_type == 'undirected':
    init_G = nx.from_edgelist(edge_list_init)
    edge_list = list(init_G.edges())
    print(len(edge_list))
elif g_type == 'directed':
    edge_list = edge_list_init

p_train, p_val, p_test = pos_partition(edge_list, g_type)
upd_list_v, n_val = neg_set(edge_list, node_list, len(p_val), g_type)
upd_list_t, n_test = neg_set(upd_list_v, node_list, len(p_test), g_type)

G = nx.from_edgelist(p_train)
max_node = max(node_list)+1
adj_matrix = build_adj_list(p_train, max_node)
num_classes = max(Y)+1
print(num_classes)

if num_classes != 3:
    raise ValueError(f"Error: Invalid number of classes for this code - Update to {num_classes} classes")

database(name, -1, G, p_train, adj_matrix, num_classes, g_type, F)
database(name, -2, G, p_val, adj_matrix, num_classes, g_type, F)
database(name, -3, G, p_test, adj_matrix, num_classes, g_type, F)
database(name, -4, G, n_val, adj_matrix, num_classes, g_type, F)
database(name, -5, G, n_test, adj_matrix, num_classes, g_type, F)

for i in range(10):
    _, n_train = neg_set(upd_list_t, node_list, len(p_train), g_type)
    database(name, i, G, n_train, adj_matrix, num_classes, g_type, F)


end_time = time.time()
print(end_time - init_time)
