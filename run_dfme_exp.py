import argparse

from attack_dfme import DFMEAttack
from defense import MSDefense
import common as comm


def run_dfme(args):

    msd = MSDefense(args)

    if args.dataset == 'MNIST':
        msd.load(netv_path='saved_model/pretrained_net/net3conv_mnist.pth')
    else:
        return

    msa = DFMEAttack(args, defense_obj=msd)
    msa.load()

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    msa.dfme_train_netS('saved_model/dfme_netS_%s' % (args.dataset),
                   'saved_model/dfme_netG_%s' % (args.dataset))

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    msa.attack("FGSM")


def get_args(dataset, cuda, expt="attack", l_only=False):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=cuda, action='store_true', help='using cuda')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_itrs', type=int, default=50)
    args.add_argument('--epoch_dg_s', type=int, default=5, help='for training net G')
    args.add_argument('--epoch_dg_g', type=int, default=1, help='for training net S')

    args.add_argument('--z_dim', type=int, default=128, help='the dimension of noise')
    args.add_argument('--batch_size_g', type=int, default=256, help='the batch size of training data')

    if dataset == "mnist":
        args.add_argument('--epoch_dg', type=int, default=50, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.0001)
        args.add_argument('--steps', nargs='+', default=[0.1, 0.5, 0.9], type=float)
        args.add_argument('--scale', type=float, default=3e-1)

        args.add_argument('--dataset', type=str, default='MNIST')
        args.add_argument('--res_filename', type=str, default='mnist_dfme')
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = get_args(dataset="mnist", cuda=False, expt="attack", l_only=False)
    print(args)

    print("Data free model stealing experiments:")
    run_dfme(args)