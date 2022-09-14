# neuralfieldDBSmodel

This repository contains the source code of the article [1].

The repository is organized as follows:

+ **src** : Contains the source codes for all experimental protocols 
    described in [1]. 
+ **tools**: This folder contains tools for visualization.

+ **params**: Here are all the configuration files containing the parameters 
    for each experimental protocol.


### Dependencies
  - Numpy
  - Matplotlib
  - Scipy


### Platforms where the code has been tested
  - Ubuntu 20.04.5 LTS
    - GCC 9.4.0
    - Python 3.8.10
    - x86_64

### Caveats

> Before run the scripts please make sure all the paths found within the scripts
  are compatible with your own system.

> If you'd like to reproduce figures 7 and 10 from [1], you would have to manually
  change the value of K_c in the file *params/params_protocolA.cfg*. The same
  value used for K_c should be also placed as name of the saved files in the scripts
  *protocoEfficiency.py* line 247 and *protocolDelays* line 231. 
  

### References
G. Is. Detorakis, A. Chaillet, S. Palfi, and S. Senova, *Closed-loop stimulation of a delayed neural fields model of parkinsonian STN-GPe network: a theoretical and computational study*,
Frontiers in neuroscience, 2015.

