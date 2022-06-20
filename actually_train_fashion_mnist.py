import argparse

from defense import MSDefense

def get_args(dataset, cuda):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=cuda, action='store_true', help='using cuda')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_itrs', type=int, default=50)
    args.add_argument('--epoch_dg_s', type=int, default=5, help='for training net G')
    args.add_argument('--epoch_dg_g', type=int, default=1, help='for training net S')

    args.add_argument('--z_dim', type=int, default=128, help='the dimension of noise')
    args.add_argument('--batch_size_g', type=int, default=25, help='the batch size of training data')

    args.add_argument('--target_accuracy', type=int, default=90, help='The accuracy of the student model which we will '
                                                                      'stop the attack from running at')

    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--epoch_b', type=int, default=10)

    if dataset == "mnist":
        args.add_argument('--epoch_dg', type=int, default=50, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.00375)
        args.add_argument('--lr_tune_g', type=float, default=0.0001)
        args.add_argument('--steps', nargs='+', default=[0.1, 0.5, 0.9], type=float)
        args.add_argument('--scale', type=float, default=3e-1)

        args.add_argument('--dataset', type=str, default='MNIST')
        args.add_argument('--res_filename', type=str, default='mnist_dfme')

    elif dataset == "fashion_mnist":
        args.add_argument('--epoch_dg', type=int, default=50, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.0001)
        args.add_argument('--steps', nargs='+', default=[0.1, 0.5, 0.9], type=float)
        args.add_argument('--scale', type=float, default=3e-1)

        args.add_argument('--dataset', type=str, default='FASHIONMNIST')
        args.add_argument('--res_filename', type=str, default='fashion_mnist_dfme')
    args = args.parse_args()
    return args


def train_fashion_mnist(args):
    msd = MSDefense(args)
    msd.load()
    msd.train_netV("saved_model/pretrained_net/trained_fashion_mnist.pth")

if __name__ == '__main__':
    print("arg")
    args = get_args(dataset="fashion_mnist", cuda=False)
    print(args)

    print("training:")
    train_fashion_mnist(args)
