## Prerequisites
```
$ conda env create -f py39.yaml
$ pip install cl-gym
```
For more information of cl-gym library, you can refer the [imirzadeh/CL-Gym](https://github.com/imirzadeh/CL-Gym).

## Run
1. __MNIST__ experiment
```
$ bash scripts/script_MNIST.sh
```

2. __FashionMNIST__ experiment

```
$ bash scripts/script_FashionMNIST.sh
```

3. __BiasedMNIST__ experiment

```
$ bash scripts/script_BiasedMNIST.sh
```

4. __Drug__ experiment

```
$ bash scripts/script_Drug.sh
```
### License for Optimization Solver
Both MOSEK and CPLEX optimization packages are free for students and academics. Installing these solvers is straightforward, as you can simply follow the provided guidelines for each package.
```python
# MOSEK
https://www.mosek.com/products/academic-licenses/
https://www.mosek.com/downloads/

# CPLEX
https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
```
