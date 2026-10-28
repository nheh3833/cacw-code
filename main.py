"""Entry point: runs the federated learning experiments.

Importing federated_learning seeds the RNGs and loads the MNIST dataset
at module load time, matching the behavior of the original single-file
script.
"""
import socket
import getpass
from pathlib import Path

from federated_learning import (
    experiment_rel_mnist,
    experiment_rot_mnist,
    experiment_cifar100,
    experiment_cifar100_label_switch,
    experiment_leaf_shakespeare,
)


def main():
    # experiment_cifar_check_var()
    # experiment_rel_cifar100()
    # experiment_rot_mnist()

    print("Hostname:", socket.gethostname())
    print("User:", getpass.getuser())

    # This will give you "exp_teza"
    print(Path(__file__).stem)


    # The six heterogeneity settings reported in Table 1 of the paper.
    # Uncomment the ones you want to run.
    print("run remote")
    experiment_rel_mnist(0)          # mnist relabel
    experiment_rot_mnist(0)          # mnist rotate (89 clients, 4 deg steps)
    #experiment_leaf_shakespeare(15,7,0.1,1e-4)     # shakespeare n=15 m=7
    #experiment_leaf_shakespeare(30,3,0.1,1e-4)     # shakespeare n=30 m=3
    #experiment_leaf_shakespeare(200,5,0.1,1e-4)    # shakespeare n=200 m=5
    #experiment_cifar100(0.01,0.1,1e-4)        # cif
    #
    # ar100 sigma=0.01
    #experiment_cifar100(10,0.1,1e-4)          # cifar100 sigma=10
    #experiment_cifar100_label_switch(0,0.1,1e-4)   # cifar100 label switch



if __name__ == "__main__":
    main()
