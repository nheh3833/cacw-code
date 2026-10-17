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


    print("run remote")
    experiment_rel_mnist()
    experiment_rot_mnist()
    #experiment_leaf_shakespeare(15,7)
    #experiment_leaf_shakespeare(30,3)
    #experiment_leaf_shakespeare(200,5)
    #experiment_cifar100(0.1)
    #experiment_cifar100(10)
    #experiment_cifar100_label_switch(0)



if __name__ == "__main__":
    main()
