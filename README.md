# Comparing the MoTEF Federated Learning Algorithm to SOTA
In this course work project we will benchmark the [MoTEF algorithm](https://arxiv.org/pdf/2405.20114) 
proposed by Islamov et al. and compare the measurements with another state of the art federated learning 
algorithms BEER (https://users.ece.cmu.edu/~yuejiec/papers/BEER.pdf).
The main.py file is the main file of this project where parameters can be specified.
All the results are summarized in the project_report.pdf and the raw data for those results can be found in param_tests.
## Project Structure for Reproducability

### Experiment setup
We wanted to test these algorithms on a real world setup: we modelled the nodes as separate processes that indeed pass messages via shared memory, as they are all coexisting on the same physical machine.
As sharing one graphics card among multiple processes is in this setting a bad idea, we had to rely on CPUs for the training. 
In an even more real world scenario however, one would probably be able to assign each node (VM) its own GPU which greatly speeds up training (we therefore went with a small model and dataset). But for this project we were limited to train on CPUs.
We tried to keep the results/benchmarks as representable as possible, but between different experiments the accuracies and timings varied greatly.

### main.py
This is the file to run the experiments. 
At the bottom you can specify all the different options we tested and run the learning procedure with multiple combinations of parameters.

### Models
Our experiments unfortunately only work with a 2 layer MLP as of today, because ResNet models cause problems (simply "not learning") which we could not trace back to its origin.

### Optimizers
We implemented the Beer and MoTEF optimizers as classes that keep track of all the local variables. We tried to implement the classes as close to the traditional torch.optimizer class as possible.
However because this is a distributed algorithm it was difficult to completely follow the class method signitures (__init__() and step()), but we kept the names the same for easier adoption.








