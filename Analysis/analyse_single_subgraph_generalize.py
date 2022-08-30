import networkx as nx
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from numpy import mean
from bokeh.io import export_png

# save the code for Fig. 11 of the paper for showing the distributions

def make_plot(title, hist, edges, xaxis_title, yaxis_title):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = xaxis_title
    #p.yaxis.axis_label = yaxis_title
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.axis_label_text_color = "black"
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_color= "black"
    p.xaxis.major_label_text_color = "black"
    p.grid.grid_line_color="white"
    return p


# path_dir = 'Result'
# results = []
# for f in os.listdir(path_dir):
#     if f.endswith('ps_var.gpickle'):
#         results.append(f)

#result = results[0]
# original graph of each utility
G = nx.read_gpickle('Result/0_ps_var.gpickle')
soln_size = 0
sec_zones_per_util = []
#fname = result.split('.')[0]
res = scipy.io.loadmat('Result/Utility_0_size37_ps200.mat')
solns = res['soln']
print('\n')
#print('************Solution for Utility '+fname+'=>')
print('Graph size : ' + str(len(G.nodes)))
nbins = 15
# if int(len(G.nodes)/3) > nbins:
#     nbins = int(len(G.nodes)/3)


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
    #print('Firewalls: '+str(list(res['score'])[k][0]) + ', ACLs: '+str(list(res['score'])[k][1]) +', Security Zones: '+str(len(SG)))
    print('Firewalls: ' + str(list(res['score'])[k][0]) + ', ACLs: ' + str(list(res['score'])[k][1]) +', Phy Sec: '+str(1000/list(res['score'])[k][2])+
          ',LODF score: '+str(1/list(res['score'])[k][3])+', Security Zones: '+str(len(SG)))
    sec_zones_per_util.append(len(SG))

print('Avg sec zone '+str(mean(sec_zones_per_util)))
# Subnet Distribution

sz = np.linspace(min(sec_zones_per_util) - 1, max(sec_zones_per_util) + 1, nbins)
hist4, edges4 = np.histogram(sec_zones_per_util, density=False, bins=sz.tolist())
p4 = make_plot("No. of Solns. Distribution", hist4, edges4, 'Security zones','Distribution')

bin_indexes ={}
# list of dictionary to score objective values
F = [{} for _ in range(4)]

for k, soln in enumerate(solns):
    edge_remove_index = np.where(np.array(soln) == 1)
    edge_list = list(G.edges)
    temp_graph = G.copy()
    for i, edge in enumerate(edge_list):
        if i in list(edge_remove_index[0]):
            temp_graph.remove_edge(edge[0], edge[1])
    SG = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
    for h in range(nbins - 1):
        if (len(SG) >= edges4[h] and len(SG) <= edges4[h+1]):
            if h not in bin_indexes.keys():
                bin_indexes[h] = []
                for f in F:
                    f[h] = []
            bin_indexes[h].append(k)
            for m,f in enumerate(F):
                if m == 2: # Security Index
                    f[h].append(list(1000/res['score'])[k][m])
                elif m == 3: # LODF index
                    f[h].append(list(1 / res['score'])[k][m])
                else:
                    f[h].append(list(res['score'])[k][m])

f_avg = []
for i,va in enumerate(F):
    dummy = []
    for k,v in va.items():
        dummy.append(mean(v))
    f_avg.append(dummy)

score_plots = [p4]
score_texts = ['FW Distribution','ACL Distribution','SI Distribution','LODF Distribution']
#score_texts = ['ACL distribution','SI distribution','LODF distribution']
for r, u in enumerate(f_avg):
    score_plots.append(make_plot(score_texts[r],u,edges4,'Security zones',score_texts[r]))

#export_png(gridplot(score_plots, ncols=5, width=300, height=300, toolbar_location=None), filename='util0_distrib.png')
show(gridplot(score_plots, ncols=5, width=300, height=300, toolbar_location=None))




