# GAMMA-TimeLoop #
This is the implementation of the mapper [GAMMA](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/08/gamma_iccad2020.pdf) for Timeloop.


### Setup Timeloop ###
Install and set up Timeloop. For more setup detail, please follow the setup user guide in [Timeloop](https://github.com/NVlabs/timeloop/blob/master/README.md).

Install the following dependencies.
```
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
libyaml-cpp-dev
libncurses-dev
libtinfo-dev
libgpm-dev
```

Clone the timeloop repository.
```
git clone ssh://path/to/timeloop.git
```

Place a symbolic link to the pat model like so:
```
cd timeloop/src
ln -s ../pat-public/src/pat .
cd ..
```

Install [Accelergy](http://accelergy.mit.edu). For more setup detail, please follow the setup user guide in [Accelergy](http://accelergy.mit.edu).
```
git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
pip install .
```

Build Timeloop
```
scons --accelergy
```

Setup path for Timeloop
```
source [path to timeloop]/timeloop/env/setup-env.bash
export PATH="$PATH:[path to timeloop]/timeloop/build"
export LIBTIMELOOP_PATH="[path to timeloop]/timeloop"
```

### Setup Timeloop-python ###
[Timeloop-python](https://github.com/Accelergy-Project/timeloop-python/tree/main/pytimeloop) is a python interface for timeloop.
For relaxing the issue of backward compatibility in the future, we clone a version of Timeloop-python in this repo.
We suggest using the cloned version here for reducing potential issues.

Install [Timeloop-python](https://github.com/Accelergy-Project/timeloop-python/tree/main/pytimeloop). For more setup detail, please visit [Timeloop-python](https://github.com/Accelergy-Project/timeloop-python/tree/main/pytimeloop).

Get into the timeloop-python directory
```
cd timeloop-python-clone/timeloop-python
```

Update the git submodules using
```
git submodule update --init
```

Remove old build
```
rm -rf build
```

Install it
```
pip install -e .
```


### Run ###
Run GAMMA-Timeloop
```
./run_gamma_timeloop.sh
```

Run GAMMA-Timeloop with multi-objective
```
./run_gamma_timeloop_multiObjective.sh
```

### Parameter
We support naive multi-objective optimization, where the user can specify up to three different objectives. If the user want single-objective optimization, simply don't specify fitness2 and fitness3.
* fitness1: The fitness objective 
* fitness2: (Optional) The second objective 
* fitness3: (Optional) The third objective
* arch_path: Architecture configuration path, e.g., arch.yaml
* problem_path: Problem configuration path, e.g., problem.yaml
* sparse_path: (Optional) Sparse configuration path, e.g., sparse.yaml
* epochs: Number of generations
* num_pops: Number of populations
* save_chkpt: To save the trace of improvement over epoch or not. Specify if the user want to save the trace.
* report_dir: The report directory for the generated map.yaml and the trace-file

##### To find out all the options
```
python main.py --help
```

