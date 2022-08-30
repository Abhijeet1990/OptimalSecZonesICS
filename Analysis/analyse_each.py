import networkx as nx
import os
import scipy.io
import numpy as np


res = scipy.io.loadmat('Utility_4.mat')
solns = res['soln']
G = nx.read_gpickle('4.gpickle')
for soln in solns:
    edge_remove_index = np.where(np.array(soln) == 1)
    edge_list = list(G.edges)
    temp_graph = G.copy()
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            temp_graph.remove_edge(edge[0], edge[1])
    SG = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
    arr = np.array([val for (node, val) in G.degree()])
    avg_degree = round(np.average(arr[1:]),3)
    print('Graph size : ' + str(len(G.nodes))+' Average Degree: '+str(avg_degree)+' Subgraphs : '+str(len(SG))+' Comp Time: '+str(res['time'][0][0]))
#print(res['score'])
# Show the results as ideal optimal zones for given number of original nodes

# node distribution in each zone
# computation time for the node sizes