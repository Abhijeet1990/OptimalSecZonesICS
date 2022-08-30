import networkx as nx
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

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

# load the gpickle files
with_load = True
path_dir = 'Set_without_LODF_other'
if with_load:
    path_dir = 'Result'
results = []
for f in os.listdir(path_dir):
    if f.endswith('cp_var.gpickle'):
        results.append(f)
# graph_sizes = []
# comp_times = []
# soln_sizes = []
# sec_zones = []
# firewall_dist = []
# acl_dist = []
# res_scores=[]
# lodf_scores=[]
#for result in results:
result = results[0]
# original graph of each utility
G = nx.read_gpickle('Result/0_cp_var.gpickle')
soln_size = 0
sec_zones_per_util = []
firewall_per_util = []
acl_per_utils = []
res_score_per_util = []
lodf_score_per_util = []
fname = result.split('.')[0]
res = scipy.io.loadmat('Result/Utility_0_size11_cp20.mat')
solns = res['soln']
print('\n')
print('************Solution for Utility '+fname+'=>')
print('Graph size : ' + str(len(G.nodes)))
print('Comp Time: '+str(res['time'][0][0]))
#graph_sizes.append(len(G.nodes))
#comp_times.append(res['time'][0][0])

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

    #if len(SG) > 1:
    print('Firewalls: '+str(list(res['score'])[k][0]) + ', ACLs: '+str(list(res['score'])[k][1]) + ', Phy Sec: '+str(1000/list(res['score'])[k][2])+', Security Zones: '+str(len(SG)))
    sec_zones_per_util.append(len(SG))
    firewall_per_util.append(list(res['score'])[k][0])
    acl_per_utils.append(list(res['score'])[k][1])
    res_score_per_util.append(1000/list(res['score'])[k][2])
    lodf_score_per_util.append(1/list(res['score'])[k][3])

# Subnet Distribution

sz = np.linspace(min(sec_zones_per_util) - 1, max(sec_zones_per_util) + 1, 10)
hist4, edges4 = np.histogram(sec_zones_per_util, density=False, bins=sz.tolist())
p4 = make_plot("Security Zone Size Distribution", hist4, edges4, 'Number of zones', 'distribution')

f = np.linspace(min(firewall_per_util) - 1, max(firewall_per_util) + 1, 10)
hist5, edges5 = np.histogram(firewall_per_util, density=False, bins=f.tolist())
p5 = make_plot("Firewall Distribution", hist5, edges5, 'Number of firewalls', 'distribution')

ac = np.linspace(min(acl_per_utils) - 1, max(acl_per_utils) + 1, 10)
hist6, edges6 = np.histogram(acl_per_utils, density=False, bins=ac.tolist())
p6 = make_plot("ACLs Distribution", hist6, edges6, 'Number of ACLs', 'distribution')

r = np.linspace(min(res_score_per_util)-1, max(res_score_per_util)+1, 10)
hist7, edges7 = np.histogram(res_score_per_util, density=False, bins=r.tolist())
p7 = make_plot("Resilience Metric Distribution", hist7, edges7, 'Resilience Score', 'distribution')

l = np.linspace(min(lodf_score_per_util), max(lodf_score_per_util), 10)
hist8,edges8 = np.histogram(lodf_score_per_util, density=False, bins=l.tolist())
p8 = make_plot("LODF Distribution", hist8, edges8, 'LODF score','distribution')

show(gridplot([p4, p5, p6, p7, p8], ncols=3, width=400, height=400, toolbar_location=None))

    #break

    # try:
    #     sec_zones.append(sum(sec_zones_per_util)/len(sec_zones_per_util))
    #     soln_sizes.append(soln_size)
    #     firewall_dist.append(sum(firewall_per_util)/len(firewall_per_util))
    #     acl_dist.append(sum(acl_per_utils)/len(acl_per_utils))
    #
    #     # invert these two soln.
    #     res_scores.append(len(res_score_per_util)/sum(res_score_per_util))
    #     # lodf_scores.append(len(lodf_score_per_util)/sum(lodf_score_per_util))
    # except:
    #     pass
# Show the results as ideal optimal zones for given number of original nodes



