# OptimalSecZonesICS
This repository propose a multi-objective optimization problem for firewall modeling for threat-resilient micro-segmentation in power system networks and solve the problem using Non-dominated Sorting Genetic Algorithm - 2 (NSGA-2). We formulate a method to find optimal network clusters considering hybrid utility network topologies where the resulting electronic security perimeters can reduce cyber investment as well as maintain grid resilience.

**Formulate four objective function**
- *Minimization of Firewall Resource*
- *Minimize the number of Access Control Lists*
- *Maximize Physical Resilience Metric*
- *Minimize the Line Outage Distribution Factor*

**List of Constraints**
- Number of clusters within a subgraph has a upper and lower limit.
- Lower limit on number of nodes within a cluster
- There exist one node in a cluster that has a *UCC* node as its neighbor in the original graph.

**Description of the files and folders within `Problems` folder**
- `OptimalZone.py` : This is the child class of the Problem class in the *pymoo* model class. This class defines the objectives and constraints of the Optimal Network Clustering problem. The `compute_constraint.py` function defines the constraints. While `compute_metric_cyber_infrastructure()` and `compute_si()` functions compute the objective functions w.r.t cyber infrastructure and physical metric respectively. 
- `OptimalZone_ObjectiveImpact.py` : This is similar to `OptimalZone.py` except it provides mechanism to select a set of objective under consideration. 

**Description of the files and folders within `Solutions` folder**
- `Compare_GA_parameter.py` : This program solves the clustering problem by comparing various Genetic Algorithm parameters such as initial population size, crossover probability, mutation probability etc. The `compute_lodf()` function computes the LODF by using the SimAuto object of *PowerWorld*. The `compute_complete_graph()` computes the complete graph by joining each sub-graph formed using a utility and their respective substations connected in a hybrid topology. `compute_graph_with_lodf()` computes the complete data structure with lodf and network topology information configured for solving the optimization problem. 
- `Compare_Objective_Impact.py` : This program solves the optimization problem by considering different set of objective functions by utilizing the class from `OptimalZone_ObjectiveImpact.py`.

**Description of the files and folders within `Analysis` folder**
This folder contains the analysis of the results obtained in the `Results` folder.

**TX2000** contains the *PowerWorld* file for the Texas synthetic grid with 2000 bus. While `SubstationGeoAll.csv` contains the geographical location information of the substations in the Texas electric grid. `SubstationRelayCount.csv` contains the information of the number of relays deployed in each substation.