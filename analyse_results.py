import networkx as nx
import os
import scipy.io
import numpy as np

# load the gpickle files
with_load = True
path_dir = 'Set3_without_LODF'
if with_load:
    path_dir = 'Set_LODF_other'
results = []
for f in os.listdir(path_dir):
    if f.endswith('.gpickle'):
        results.append(f)

for result in results:
    # original graph of each utility
    G = nx.read_gpickle(path_dir+'/'+result)

    fname = result.split('.')[0]
    res = scipy.io.loadmat(path_dir+'/'+'Utility_'+fname+'.mat')
    solns = res['soln']
    print('\n')
    print('************Solution for Utility '+fname+'=>')
    print('Graph size : ' + str(len(G.nodes)))
    print('Comp Time: '+str(res['time'][0][0]))
    for k, soln in enumerate(solns):
        edge_remove_index = np.where(np.array(soln) == 1)
        edge_list = list(G.edges)
        temp_graph = G.copy()
        for i, edge in enumerate(edge_list):
            if i in list(edge_remove_index[0]):
                temp_graph.remove_edge(edge[0], edge[1])
        SG = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
        arr = np.array([val for (node, val) in G.degree()])
        avg_degree = round(np.average(arr[1:]),3)

        if len(SG) > 1:
            print('Firewalls: '+str(list(res['score'])[k][0]) + ', ACLs: '+str(list(res['score'])[k][1]) + ', Phy Sec: '+str(list(res['score'])[k][2])+ ', LODF : '+str(list(res['score'])[k][3])+', Security Zones: '+str(len(SG)))
# Show the results as ideal optimal zones for given number of original nodes

# node distribution in each zone
# computation time for the node sizes