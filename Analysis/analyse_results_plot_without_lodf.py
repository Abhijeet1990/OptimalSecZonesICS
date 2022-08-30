import networkx as nx
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show


# load the gpickle files
with_load = False
path_dir = 'Set_without_LODF_other'
if with_load:
    path_dir = 'Set2_LODF_other'
results = []
for f in os.listdir(path_dir):
    if f.endswith('.gpickle'):
        results.append(f)
graph_sizes = []
comp_times = []
soln_sizes = []
sec_zones = []
firewall_dist = []
acl_dist = []
res_scores=[]
lodf_scores=[]
for result in results:
    # original graph of each utility
    G = nx.read_gpickle(path_dir+'/'+result)
    soln_size = 0
    sec_zones_per_util = []
    firewall_per_util = []
    acl_per_utils = []
    res_score_per_util = []
    lodf_score_per_util = []
    fname = result.split('.')[0]
    res = scipy.io.loadmat(path_dir+'/'+'Utility_'+fname+'.mat')
    solns = res['soln']
    print('\n')
    print('************Solution for Utility '+fname+'=>')
    print('Graph size : ' + str(len(G.nodes)))
    print('Comp Time: '+str(res['time'][0][0]))
    graph_sizes.append(len(G.nodes))
    comp_times.append(res['time'][0][0])
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
            soln_size += 1
            print('Firewalls: '+str(list(res['score'])[k][0]) + ', ACLs: '+str(list(res['score'])[k][1]) + ', Phy Sec: '+str(list(res['score'])[k][2])+', Security Zones: '+str(len(SG)))
            sec_zones_per_util.append(len(SG))
            firewall_per_util.append(list(res['score'])[k][0])
            acl_per_utils.append(list(res['score'])[k][1])
            res_score_per_util.append(list(res['score'])[k][2])
            # lodf_score_per_util.append(list(res['score'])[k][3])

    try:
        sec_zones.append(sum(sec_zones_per_util)/len(sec_zones_per_util))
        soln_sizes.append(soln_size)
        firewall_dist.append(sum(firewall_per_util)/len(firewall_per_util))
        acl_dist.append(sum(acl_per_utils)/len(acl_per_utils))

        # invert these two soln.
        res_scores.append(len(res_score_per_util)/sum(res_score_per_util))
        # lodf_scores.append(len(lodf_score_per_util)/sum(lodf_score_per_util))
    except:
        pass
# Show the results as ideal optimal zones for given number of original nodes


def make_plot(title, hist, edges, xaxis_title, yaxis_title):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = xaxis_title
    p.yaxis.axis_label = yaxis_title
    p.grid.grid_line_color="white"
    return p

# Subnet Distribution
hist, edges = np.histogram(graph_sizes, density=False, bins=[0,5,10,15,20,25,30,35,40,45,50])

x = np.linspace(0, 50, 11)
p1 = make_plot("Network Size Distribution", hist, edges, 'subnet size','distribution')


t = np.linspace(min(comp_times), max(comp_times), 10)
hist2,edges2 = np.histogram(comp_times, density=False, bins=t.tolist())
p2 = make_plot("Computation Time Distribution", hist2, edges2, 'Computation Time (in seconds)','distribution')

s = np.linspace(min(soln_sizes)- 1, max(soln_sizes)+1, 10)
hist3,edges3 = np.histogram(soln_sizes, density=False, bins=s.tolist())
p3 = make_plot("Solution Size Distribution", hist3, edges3, 'Number of Solutions','distribution')

sz = np.linspace(min(sec_zones)- 1, max(sec_zones_per_util)+1, 10)
hist4,edges4 = np.histogram(sec_zones, density=False, bins=sz.tolist())
p4 = make_plot("Security Zone Size Distribution", hist4, edges4, 'Number of zones','distribution')

f = np.linspace(min(firewall_dist)- 1, max(firewall_dist)+1, 10)
hist5,edges5 = np.histogram(firewall_dist, density=False, bins=f.tolist())
p5 = make_plot("Firewall Distribution", hist5, edges5, 'Number of firewalls','distribution')

ac = np.linspace(min(acl_dist)- 1, max(acl_dist)+1, 10)
hist6,edges6 = np.histogram(acl_dist, density=False, bins=ac.tolist())
p6 = make_plot("ACLs Distribution", hist6, edges6, 'Number of ACLs','distribution')

r = np.linspace(min(res_scores), max(res_scores), 10)
hist7,edges7 = np.histogram(res_scores, density=False, bins=r.tolist())
p7 = make_plot("Resilience Metric Distribution", hist7, edges7, 'Resilience Score','distribution')

# l = np.linspace(min(lodf_scores)- 1, max(lodf_scores)+1, 10)
# hist8,edges8 = np.histogram(lodf_scores, density=False, bins=l.tolist())
# p8 = make_plot("LODF Distribution", hist8, edges8, 'LODF score','distribution')

show(gridplot([p1,p2,p3,p4,p5,p6,p7], ncols=4, width=400, height=400, toolbar_location=None))



# node distribution in each zone
# computation time for the node sizes