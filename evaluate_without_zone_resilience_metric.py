import networkx as nx
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
import pandas as pd
import json

# figure for the last fig of the paper (fig 15)

def get_relays(substation_name):
    full_list = os.listdir('TX2000/Substations/')
    file = [nm for nm in full_list if substation_name in nm][0]
    # parse the file based on the substation name
    f = open('TX2000/Substations/' + file)
    data = json.load(f)
    vals = data['nodes']['$values']

    relay_info = {}
    if 'Line' not in relay_info.keys():
        relay_info['Line'] = 0
    if 'Disconnect' not in relay_info.keys():
        relay_info['Disconnect'] = 0
    if 'Breaker' not in relay_info.keys():
        relay_info['Breaker'] = 0
    if 'Transformer' not in relay_info.keys():
        relay_info['Transformer'] = 0
    for val in vals:
        if 'Relay' in val['$type'] and 'RelayController' not in val['$type']:
            # print(val)
            controls = val['breakers']['$values']
            for control in controls:

                if control['controlType'] == 'Line':
                    relay_info['Line'] += 1
                if control['controlType'] == 'Disconnect':
                    relay_info['Disconnect'] += 1
                if control['controlType'] == 'Breaker':
                    relay_info['Breaker'] += 1
                if control['controlType'] == 'Transformer':
                    relay_info['Transformer'] += 1
    return relay_info


def make_plot(title, hist, edges, hist2, xaxis_title, yaxis_title):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="red", line_color="white", alpha=0.5, legend_label='Mesh Topo Original')
    p.quad(top=hist2, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5, legend_label='Mesh Topo Proposed')
    p.y_range.start = 0
    p.legend.background_fill_color = "#fefefe"
    # display legend in top left corner (default is top right corner)
    p.legend.location = "top_right"
    # add a title to your legend
    p.legend.title = "Comparison"
    # change appearance of legend text
    p.legend.label_text_font = "times"
    p.legend.label_text_font_style = "italic"
    p.legend.label_text_color = "black"
    p.legend.label_text_font_style="bold"
    p.legend.label_text_font = "13pt"

    # change border and background of legend
    p.legend.border_line_width = 3
    p.legend.border_line_color = "black"
    p.legend.border_line_alpha = 0.8
    p.legend.background_fill_color = "black"
    p.legend.background_fill_alpha = 0.2

    p.xaxis.axis_label = xaxis_title
    #p.yaxis.axis_label = yaxis_title
    p.xaxis.major_label_text_font_size = "13pt"
    p.yaxis.major_label_text_font_size = "13pt"
    p.xaxis.axis_label_text_color = "black"
    p.grid.grid_line_color="white"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.xaxis.major_label_text_color = "black"
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'
    p.title.text_font_size='14pt'
    return p

# load the gpickle files
with_load = True
path_dir = 'Set3_without_LODF'
if with_load:
    path_dir = 'Set_150'
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
    G = nx.read_gpickle(path_dir + '/' + result)
    soln_size = 0
    sec_zones_per_util = []
    firewall_per_util = []
    acl_per_utils = []
    res_score_per_util = []
    lodf_score_per_util = []
    fname = result.split('.')[0]
    res = scipy.io.loadmat(path_dir+'/'+'Utility_'+fname.split('_')[0]+'.mat')
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
            print('Firewalls: '+str(list(res['score'])[k][0]) + ', ACLs: '+str(list(res['score'])[k][1]) + ', Phy Sec: '+str(list(res['score'])[k][2])+ ', LODF : '+str(list(res['score'])[k][3])+', Security Zones: '+str(len(SG)))
            sec_zones_per_util.append(len(SG))
            firewall_per_util.append(list(res['score'])[k][0])
            acl_per_utils.append(list(res['score'])[k][1])
            res_score_per_util.append(1000/list(res['score'])[k][2])
            lodf_score_per_util.append(list(res['score'])[k][3])



    try:
        sec_zones.append(sum(sec_zones_per_util)/len(sec_zones_per_util))
        soln_sizes.append(soln_size)
        firewall_dist.append(sum(firewall_per_util)/len(firewall_per_util))
        acl_dist.append(sum(acl_per_utils)/len(acl_per_utils))

        # invert these two soln.
        res_scores.append(len(res_score_per_util)/sum(res_score_per_util))
        lodf_scores.append(len(lodf_score_per_util)/sum(lodf_score_per_util))
    except:
        pass

r2 = np.linspace(min(res_scores), max(res_scores), 10)
hist8,edges8 = np.histogram(res_scores, density=False, bins=r2.tolist())


data = pd.read_csv('TX2000/SubstationGeoAll.csv')
# data
# hypothetically assign utility control center location
subname_id = {}

location = []
substationsList = []
location_dict = {}
for i, c in data.iterrows():
    location.append([c['Longitude'], c['Latitude']])
    substationsList.append(c['SubID'])
    if str(c['SubID']) in location_dict.keys():
        location_dict[str(c['SubID'])].append([c['Longitude'], c['Latitude']])
        # subname_id[str(c['SubID'])].append(str(c['SubName']))
    else:
        location_dict[str(c['SubID'])] = [[c['Longitude'], c['Latitude']]]
        subname_id[str(c['SubID'])] = str(c['SubName'])

si_g_complete = []
# for each graph obtain the resilience metric and store
for result in results:
    # original graph of each utility
    G = nx.read_gpickle(path_dir+'/'+result)
    relay_info = {}
    for node in G.nodes():
        si_g = 0
        if 'Line' not in relay_info.keys():
            relay_info['Line'] = 0
        if 'Disconnect' not in relay_info.keys():
            relay_info['Disconnect'] = 0
        if 'Breaker' not in relay_info.keys():
            relay_info['Breaker'] = 0
        if 'Transformer' not in relay_info.keys():
            relay_info['Transformer'] = 0
        if node in subname_id.keys():
            sub_relay_info = get_relays(subname_id[node])
            relay_info['Line'] = sub_relay_info['Line']
            relay_info['Disconnect'] = sub_relay_info['Disconnect']
            relay_info['Breaker'] = sub_relay_info['Breaker']
            relay_info['Transformer'] = sub_relay_info['Transformer']
        if relay_info['Breaker'] != 0:
            si_g += (1 / (1 * relay_info['Breaker']))
        if relay_info['Disconnect'] != 0:
            si_g += (1 / (3 * relay_info['Disconnect']))
        if relay_info['Transformer'] != 0:
            si_g += (1 / (4 * relay_info['Transformer']))
        if relay_info['Line'] != 0:
            si_g += (1 / (2 * relay_info['Line']))
    si_g_complete.append(si_g)

r = np.linspace(min(si_g_complete), max(res_scores), len(r2.tolist()))
hist7,edges7 = np.histogram(si_g_complete, density=False, bins=r.tolist())

p7 = make_plot("Resilience Metric Distribution", hist7, edges8, hist8,'Resilience Score','Frequency')

show(gridplot([p7], ncols=4, width=400, height=400, toolbar_location=None))