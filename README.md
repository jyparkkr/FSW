# Fair Class-Incremental Learning using Sample Weighting

## Prerequisites
```
$ conda env create -f py39.yaml
$ pip install cl-gym
```
For more information on the cl-gym library, refer to [imirzadeh/CL-Gym](https://github.com/imirzadeh/CL-Gym).

## Run
1. __MNIST__ experiment
```
$ bash scripts/MNIST_EER.sh
```

2. __FashionMNIST__ experiment

```
$ bash scripts/FashionMNIST_EER.sh
```

3. __BiasedMNIST__ experiment

```
$ bash scripts/BiasedMNIST_EO.sh
$ bash scripts/BiasedMNIST_DP.sh
```

4. __Drug__ experiment

```
$ bash scripts/script_Drug_EO.sh
$ bash scripts/script_Drug_DP.sh
```

5. __BiasBios__ experiment

```
$ bash scripts/Bios_EO.sh
$ bash scripts/Bios_DP.sh
```

## License for Optimization Solver
Both MOSEK and CPLEX optimization solvers are free for students and academics. Installing these solvers is straightforward, as you can simply follow the provided guidelines for each package.
```python
# MOSEK
https://www.mosek.com/products/academic-licenses/
https://www.mosek.com/downloads/

# CPLEX
https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
```