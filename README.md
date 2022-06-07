# GAMMA-TimeLoop #
This is the implementation of the mapper [GAMMA](https://dl.acm.org/doi/10.1145/3400302.3415639) for Timeloop.
![GAMMA Framework](./gamma.jpg)
## Install Dependency ###
### Install Timeloop ###
Timeloop documentation is hosted at [https://timeloop.csail.mit.edu/timeloop](https://timeloop.csail.mit.edu/timeloop). The guides there cover detailed installation steps. 

### Install Timeloop-python ###
```
python build_pytimeloop.py
```
For more installation detail, please visit [https://timeloop.csail.mit.edu/timeloop](https://timeloop.csail.mit.edu/timeloop).

------------------


### Take a Trial Run ###
Run GAMMA-Timeloop
```
./run_gamma_timeloop.sh
```

Run GAMMA-Timeloop with multi-objective
```
./run_gamma_timeloop_multiObjective.sh
```

For more detail, please look at [``./src``](./src)


### Citation ###
```
@inproceedings{gamma,
    author       = {Kao, Sheng-Chun and Krishna, Tushar},
    title        = {GAMMA: Automating the HW Mapping of DNN Models on Accelerators via Genetic Algorithm},
    booktitle     = {ICCAD},
  year          = {2020}
}

```
```
@inproceedings{digamma,
title={DiGamma: Domain-aware Genetic Algorithm for HW-Mapping Co-optimization for DNN Accelerators},
author={Kao, Sheng-Chun and Pellauer, Michael and Parashar, Angshuman and Krishna, Tushar},
booktitle     = {DATE},
year={2022}
}
```
```
@inproceedings{kao2022formalism,
  title={A Formalism of DNN Accelerator Flexibility},
  author={Kao, Sheng-Chun and Kwon, Hyoukjun and Pellauer, Michael and Parashar, Angshuman and Krishna, Tushar},
  booktitle={Proceedings of the 2022 ACM SIGMETRICS/IFIP PERFORMANCE Joint International Conference on Measurement and Modeling of Computer Systems},
  year={2022}
}
```