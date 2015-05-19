neuralfieldDBSmodel
===================

# 

This is a Python implementation of a Basal Ganglia model. We modeled the
**Subthalamic nucleus** (STN) and the **Globus Pallidus external** (GPe) in
order to study pathological oscillations in _Parkinson's disease_ (PD). 

The underlying model is based on Delayed Neural Fields equations and all the
details about this model can be found at: 


The repository is organized as follows:

+ **src** : Contains the source codes for all experimental protocols 
    described in (). The source code is written in Python and it takes
    advantage of Numpy and Matplotlib. In order to run a simulation you have
    to use the file **run_protocols.py**.

+ **tools**: In this folder you can find some tools for visualization. With 
    **visualization.py** you can reproduce the figures found in ().

+ **data**: Here are saved all the results once you run a simulation. 

+ **params**: Here you can find all the configuration files containing all
    the parameters (for each experimental protocol). 


Caveats
--------
> You have to make sure that all the paths found within scripts are compatible
  with your own system. The source code was mainly built on Linux (Debian)
  platform. If you would like to run specific experiments you have to properly
  modify the source code.


License
-------
You should have received a copy of the BSD 3-Clause License along with this
program. If not, see <http://www.gnu.org/licenses/>.


Bugs
----
Please report **bugs** to : <gdetor@protonmail.ch>
