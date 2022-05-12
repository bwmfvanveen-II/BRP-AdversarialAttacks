import sys
import argparse

from common import Logger
from attack import MSAttack
from defense import MSDefense
import common as comm


def run_attack(args):
    msd = MSDefense(args)
    msd.load(netv_path='saved_model/pretrained_net/net3conv_mnist.pth')

    msa = MSAttack(args, defense_obj=msd)
    msa.load()

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader)

    msa.train_netS('saved_model/netS_mnist_temp.pth', data_type="REAL", label_only=False)

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader)

    msa.attack("FGSM")
    # msa.attack("BIM")
    # msa.attack("CW")
    # msa.attack("PGD")


if __name__ == '__main__':
    sys.stdout = Logger('ms_attack.log', sys.stdout)

    args = argparse.ArgumentParser()
    args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    args.add_argument('--dataset', type=str, default='MNIST')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_b', type=int, default=20, help='for training net V')
    args.add_argument('--epoch_g', type=int, default=5, help='for training net S')

    args.add_argument('--lr', type=float, default=0.0001)
    args = args.parse_args()

    run_attack(args)


