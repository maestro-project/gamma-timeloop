# GAMMA-TimeLoop #
This is the implementation of the mapper [GAMMA](https://dl.acm.org/doi/10.1145/3400302.3415639) for Timeloop.
![GAMMA Framework](./gamma.jpg)

### Sister Repo: Gamma-Maestro ###
We also have GAMMA supporting MAESTRO as cost model. The sister repo can be found here [Gamma-Maestro](https://github.com/maestro-project/gamma). It searches through the design space of MAESTRO and proposes an optimized mapping.

----
## Install Dependency ###
### Install Timeloop ###
Timeloop documentation is hosted at [https://timeloop.csail.mit.edu/timeloop](https://timeloop.csail.mit.edu/timeloop). The guides there cover detailed installation steps. 
### Install Timeloop-python ###

```
python build_pytimeloop.py
```
For more installation detail, please visit [https://timeloop.csail.mit.edu/timeloop](https://timeloop.csail.mit.edu/timeloop).

------------------
## Take a Trial Run ##
Run GAMMA-Timeloop
```
./run_gamma_timeloop.sh
```

Run GAMMA-Timeloop with multi-objective
```
./run_gamma_timeloop_multiObjective.sh
```

For more detail, please look at [``./src``](./src)

--------------

## Citation ##
```
@inproceedings{gamma_timeloop,
    author       = {Kao, Sheng-Chun and Parashar, Angshuman and Tsai Po-An and Krishna, Tushar},
    title        = {Demystifying Map Space Exploration for NPUs},
    booktitle     = {IISWC},
  year          = {2022}
}

```


```
@inproceedings{gamma,
    author       = {Kao, Sheng-Chun and Krishna, Tushar},
    title        = {GAMMA: Automating the HW Mapping of DNN Models on Accelerators via Genetic Algorithm},
    booktitle     = {ICCAD},
  year          = {2020}
}

```
