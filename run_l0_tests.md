# Running the L0 tests

Cuda and nccl need to be installed to run the following tests. Depending on how your cluster works, you may need to 
do a module load as follows, however, this is cluster dependant. 

```bash
module load cuda13.0/toolkit/13.0.2
module load nccl2-cuda13.0-gcc/2.28.9
```

```bash
git clone https://github.com/gadamopoulos-saifh/milabench.git
# or via SSH:
git clone git@github.com:gadamopoulos-saifh/milabench.git

cd milabench

export MILABENCH_HF_TOKEN=xxxxx

chmod +x running_all_tests.sh
./running_all_tests.sh
```

Please return to us the ```results.csv```. 
