# timeloop-python
Python wrapper for the timeloop project.

## Dependencies
Since building PyTimeloop requires building Timeloop, dependencies of
Timeloop are also required.
```
// Timeloop dependencies
scons
libconfig++-dev
libboost-dev
libboost-iostreams-dev
libboost-serialization-dev
libyaml-cpp-dev
libncurses-dev
libtinfo-dev
libgpm-dev

// PyTimeloop dependencies
cmake
```

## Installing
Update the git submodules using
```
$ git submodule update
```
Timeloop has to be built manually using
```
$ ln -s /path/to/pat-public/src/pat src/pat
$ scons -j4 --accelergy
```
in the Timeloop directory (`lib/Timeloop`).
Then, install PyTimeloop by running
```
$ pip3 install -e .
```
If you ran `pip3 install -e .` recently, the `build/` directory has to be
cleaned by running `rm -rf build`.

### Using your own build of Timeloop
If you want to use your own build of Timeloop, you can set the environment
variable `LIBTIMELOOP_PATH` to your Timeloop directory.

For example, if you have the library in `/path/to/timeloop/` -- if you have 
built Timeloop, you can find `libtimeloop-model.so` in `/path/to/timeloop/lib`
-- you can execute the following
```
$ export LIBTIMELOOP_PATH=/path/to/timeloop
$ rm -rf build && pip3 install -e .
```

## Using Command Line Tools
After installing PyTimeloop, there are some premade Timeloop applications you
can use on the command line:
- `timeloop-model.py`

For example,
```
$ timeloop-model.py --help
usage: timeloop-model.py [-h] [--output_dir OUTPUT_DIR]
                         [--verbosity VERBOSITY]
                         configs [configs ...]

Run Timeloop given architecture, workload, and mapping.

positional arguments:
  configs               Config files to run Timeloop.

optional arguments:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory to dump output.
  --verbosity VERBOSITY
                        0 is only error; 1 adds warning; 2 is everyting.
```

## Contributing
This README is written with users as its audience, more information relevant
to the development of the project can be found in CONTRIBUTING.md.

