import argparse
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from attack_dfme import DFMEAttack
from defense import MSDefense
import common as comm


def run_dfme(args):
    msd = MSDefense(args)

    if args.dataset == 'MNIST':
        msd.load(netv_path='saved_model/pretrained_net/net3conv_mnist.pth')
    elif args.dataset == 'FASHIONMNIST':
        msd.load(netv_path='saved_model/pretrained_net/trained_fashion_mnist.pth')
    else:
        return

    # print(msd.state_dict)

    msa = DFMEAttack(args, defense_obj=msd)
    msa.load()

    print(args.cuda)

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    startTime = time.time()
    learning_rate, query_per_epoch, accumulative_queries, total_queries, num_epochs = msa.dfme_train_netS(
        'saved_model/dfme_netS_%s' % args.dataset, 'saved_model/dfme_netG_%s' % args.dataset, args.target_accuracy)
    endTime = time.time()
    timeTaken = endTime - startTime

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    attack_success_rate = msa.attack("FGSM")

    return args, learning_rate, query_per_epoch, accumulative_queries, total_queries, attack_success_rate, num_epochs, \
           timeTaken


def run_dfme_no_attack(args):
    msd = MSDefense(args)

    if args.dataset == 'MNIST':
        msd.load(netv_path='saved_model/pretrained_net/net3conv_mnist.pth')
    elif args.dataset == 'FASHIONMNIST':
        msd.load(netv_path='saved_model/pretrained_net/trained_fashion_mnist.pth')
    else:
        return

    msa = DFMEAttack(args, defense_obj=msd)
    msa.load()

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    startTime = time.time()
    learning_rate, query_per_epoch, accumulative_queries, total_queries, num_epochs = msa.dfme_train_netS(
        'saved_model/dfme_netS_%s' % args.dataset, 'saved_model/dfme_netG_%s' % args.dataset, args.target_accuracy)
    endTime = time.time()
    timeTaken = endTime - startTime

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)
    #
    # attack_success_rate = msa.attack("FGSM")

    return args, learning_rate, query_per_epoch, accumulative_queries, total_queries, num_epochs, timeTaken


def get_args(dataset, cuda):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=cuda, action='store_true', help='using cuda')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_itrs', type=int, default=50)
    args.add_argument('--epoch_dg_s', type=int, default=5, help='for training net G')
    args.add_argument('--epoch_dg_g', type=int, default=1, help='for training net S')

    args.add_argument('--z_dim', type=int, default=128, help='the dimension of noise')
    args.add_argument('--batch_size_g', type=int, default=50, help='the batch size of training data')

    args.add_argument('--target_accuracy', type=int, default=100, help='The accuracy of the student model which we will '
                                                                      'stop the attack from running at')

    if dataset == "mnist":
        args.add_argument('--epoch_dg', type=int, default=50, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.008327361955412864)
        args.add_argument('--lr_tune_g', type=float, default=0.0007222586855333449)
        args.add_argument('--steps', nargs='+', default=[0.08170446287812391, 0.6479498522799164, 0.8323928808196334], type=float)
        args.add_argument('--scale', type=float, default=0.3882951028871382)

        args.add_argument('--dataset', type=str, default='MNIST')
        args.add_argument('--res_filename', type=str, default='mnist_dfme')

    elif dataset == "fashion_mnist":
        args.add_argument('--epoch_dg', type=int, default=100, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.00857)
        args.add_argument('--lr_tune_g', type=float, default=0.000734)
        args.add_argument('--steps', nargs='+', default=[0.21, 0.6, 0.929], type=float)
        args.add_argument('--scale', type=float, default=0.3)

        args.add_argument('--dataset', type=str, default='FASHIONMNIST')
        args.add_argument('--res_filename', type=str, default='fashion_mnist_dfme')
    args = args.parse_args()
    return args


def set_args(args_to_change, epoch_dg=50, lr_tune_s=0.001, lr_tune_g=0.0001, steps=[0.1, 0.5, 0.9], scale=3e-1,
             dataset="NO_CHANGE", target_accuracy=100, batch_size=25):
    args_to_change.epoch_ds = epoch_dg
    args_to_change.lr_tune_s = lr_tune_s
    args_to_change.lr_tune_g = lr_tune_g
    args_to_change.steps = steps
    args_to_change.scale = scale

    args_to_change.target_accuracy = target_accuracy
    args_to_change.batch_size_g = batch_size

    if dataset == "mnist":
        args_to_change.dataset = "MNIST"
    elif dataset == "fashion_mnist":
        args_to_change.dataset = "FASHIONMNIST"

def args_to_dict(args):
    return vars(args)


def save_results(args, data, query_per_epoch, accumulative_queries, total_queries, attack_success_rate, num_epochs,
                 time_taken):
    epoch_numbers = np.array([i for i in range(1, num_epochs + 1)])
    epoch_word = np.array(["epoch "] * num_epochs)
    index_name = np.core.defchararray.add(epoch_word.astype(str), epoch_numbers.astype(str))

    dataframe = pd.DataFrame(np.transpose(np.array([data, query_per_epoch, accumulative_queries])),
                             columns=["accuracy", "queries made", "total queries made"], index=index_name)

    important_info = "Dataset: " + args.dataset + "\nLearning rate of S: " + str(args.lr_tune_s) + \
                     "\nLearning rate of G: " + str(args.lr_tune_g) + "\nz_dim: " + str(args.z_dim) + \
                     "\nNumber of epochs:" + str(args.epoch_dg) + "\nStep timings: " + str(args.steps) + \
                     ", and scale: " + str(args.scale) + "\nepoch_ds_S: " + str(args.epoch_dg_s) + ", and G: " + \
                     str(args.epoch_dg_g) + "\nTotal queries: " + str(total_queries) + "\nAttack success rate: " + \
                     str(attack_success_rate) + "%\nTotal time taken to run: " + str(time_taken) + "\n"

    with open('dfme_exp_results_post_optim.txt', 'a') as outputfile:
        # outputfile.write(json.dumps(vars(args.)))
        outputfile.write(important_info)
        outputfile.write(dataframe.to_string(header=True, index=True, index_names=True))
        outputfile.write("\n--------------------------------------------------------------------------"
                         "---------------------------------------------\n")


def make_graph(args, data, num_epochs):
    fig, plot = plt.subplots()
    plot.scatter(range(1, num_epochs + 1), data)

    title = "S learning rate: " + str(args.lr_tune_s) + ", G learning rate: " + str(args.lr_tune_g)
    plot.set_title(title)
    plt.xlabel("epoch number")
    plt.ylabel("accuracy of netS (%)")
    plt.show()


if __name__ == '__main__':
    args = get_args(dataset="fashion_mnist", cuda=torch.cuda.is_available())

    print("Data free model stealing experiments:")

    args_set, learning_rate, query_per_epoch, accumulative_queries, total_queries, attack_success_rate, num_epochs , time_taken = run_dfme(args)


    save_results(args_set, learning_rate, query_per_epoch, accumulative_queries, total_queries, attack_success_rate,
                 num_epochs, time_taken)
    make_graph(args_set, learning_rate, num_epochs)
