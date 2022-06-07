# GAMMA-TimeLoop #
[GAMMA: Automating the HW Mapping of DNN Models on
Accelerators via Genetic Algorithm](https://dl.acm.org/doi/10.1145/3400302.3415639)

### Parameter
We support naive multi-objective optimization, where the user can specify up to three different objectives. If the user want single-objective optimization, simply don't specify fitness2 and fitness3.
* fitness1: The fitness objective 
* fitness2: (Optional) The second objective 
* fitness3: (Optional) The third objective
* config_path: Configuration path, should include arch.yaml, problem.yaml, (and sparse.yaml if sparsity is considered)
* use_sparse: Enable it to explore sparse accelerator space, otherwise explore dense accelerator space
* explore_bypass: Enable it to explore bypass buffer option
* epochs: Number of generations
* num_pops: Number of populations
* save_chkpt: To save the trace of improvement over epoch or not. Specify if the user want to save the trace.
* report_dir: The report directory for the generated map.yaml and the trace-file



