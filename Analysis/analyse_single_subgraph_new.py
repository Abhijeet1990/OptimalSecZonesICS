import networkx as nx
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from numpy import mean

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


path_dir = 'Result'
results = []
for f in os.listdir(path_dir):
    if f.endswith('ps_var.gpickle'):
        results.append(f)

result = results[0]
# original graph of each utility
G = nx.read_gpickle('Result/0_ps_var.gpickle')
soln_size = 0
sec_zones_per_util = []
fname = result.split('.')[0]
res = scipy.io.loadmat('Result/Utility_0_size37_ps100.mat')
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

# Subnet Distribution

sz = np.linspace(min(sec_zones_per_util) - 1, max(sec_zones_per_util) + 1, 5)
hist4, edges4 = np.histogram(sec_zones_per_util, density=False, bins=sz.tolist())
p4 = make_plot("Security Zone Size Distribution", hist4, edges4, 'Number of zones', 'distribution')

bin_indexes ={}
f1= {}
f2= {}
f3= {}
f4= {}
for k, soln in enumerate(solns):
    edge_remove_index = np.where(np.array(soln) == 1)
    edge_list = list(G.edges)
    temp_graph = G.copy()
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            temp_graph.remove_edge(edge[0], edge[1])
    SG = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]

    if (len(SG) > edges4[0] and len(SG) < edges4[1]):
        if 0 not in bin_indexes.keys():
            bin_indexes[0] = []
            f1[0] = []
            f2[0] = []
            f3[0] = []
            f4[0] = []
        bin_indexes[0].append(k)
        f1[0].append(list(res['score'])[k][0])
        f2[0].append(list(res['score'])[k][1])
        f3[0].append(list(1000/res['score'])[k][2])
        f4[0].append(list(1/res['score'])[k][3])
    elif (len(SG) > edges4[1] and len(SG) < edges4[2]):
        if 1 not in bin_indexes.keys():
            bin_indexes[1] = []
            f1[1] = []
            f2[1] = []
            f3[1] = []
            f4[1] = []
        bin_indexes[1].append(k)
        f1[1].append(list(res['score'])[k][0])
        f2[1].append(list(res['score'])[k][1])
        f3[1].append(list(1000/res['score'])[k][2])
        f4[1].append(list(1/res['score'])[k][3])
    elif (len(SG) > edges4[2] and len(SG) < edges4[3]):
        if 2 not in bin_indexes.keys():
            bin_indexes[2] = []
            f1[2] = []
            f2[2] = []
            f3[2] = []
            f4[2] = []
        bin_indexes[2].append(k)
        f1[2].append(list(res['score'])[k][0])
        f2[2].append(list(res['score'])[k][1])
        f3[2].append(list(1000/res['score'])[k][2])
        f4[2].append(list(1/res['score'])[k][3])
    elif (len(SG) > edges4[3] and len(SG) < edges4[4]):
        if 3 not in bin_indexes.keys():
            bin_indexes[3] = []
            f1[3] = []
            f2[3] = []
            f3[3] = []
            f4[3] = []
        bin_indexes[3].append(k)
        f1[3].append(list(res['score'])[k][0])
        f2[3].append(list(res['score'])[k][1])
        f3[3].append(list(1000/res['score'])[k][2])
        f4[3].append(list(1/res['score'])[k][3])

f1_avg = []
f2_avg = []
f3_avg = []
f4_avg = []

for k,v in f1.items():
    f1_avg.append(mean(f1[k]))
    f2_avg.append(mean(f2[k]))
    f3_avg.append(mean(f3[k]))
    f4_avg.append(mean(f4[k]))


p5 = make_plot("Firewall Distribution", f1_avg, edges4, 'Number of zones', 'FW distribution')
p6 = make_plot("ACLs Distribution", f2_avg, edges4, 'Number of zones', 'ACL distribution')
p7 = make_plot("Resilience Metric Distribution", f3_avg, edges4, 'Number of zones', 'SI distribution')
p8 = make_plot("LODF Distribution", f4_avg, edges4, 'Number of zones','LODF distribution')

# ac = np.linspace(min(acl_per_utils) - 1, max(acl_per_utils) + 1, 10)
# hist6, edges6 = np.histogram(acl_per_utils, density=False, bins=sz.tolist())
# p6 = make_plot("ACLs Distribution", hist6, edges6, 'Number of zones', 'ACL distribution')
#
# r = np.linspace(min(res_score_per_util)-1, max(res_score_per_util)+1, 10)
# hist7, edges7 = np.histogram(res_score_per_util, density=False, bins=sz.tolist())
# p7 = make_plot("Resilience Metric Distribution", hist7, edges7, 'Number of zones', 'SI distribution')
#
# l = np.linspace(min(lodf_score_per_util), max(lodf_score_per_util), 10)
# hist8,edges8 = np.histogram(lodf_score_per_util, density=False, bins=sz.tolist())
# p8 = make_plot("LODF Distribution", hist8, edges8, 'Number of zones','LODF distribution')

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



