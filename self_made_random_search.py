import copy
import time

import torch.cuda

from run_dfme_exp import get_args, set_args, run_dfme_no_attack
from self_made_grid_search import save_best_grid_data, save_grid_data

import numpy as np


def specified_random_search(file_to_write_to="random_search_default_info_file", num_iterations=25):
    dataset = "fashion_mnist"
    args = get_args(dataset=dataset, cuda=torch.cuda.is_available())

    target_accuracy = 50

    student_lr_range = [0.00001, 0.1]
    generator_lr_range = [0.000001, 0.1]
    steps_range = [[0.01, 0.401, 0.8], [0.4, 0.799, 0.99]]
    scale_range = [0.005, 0.5]

    batch_size = 50
    best_num_epochs = 51
    best_accuracy = 0
    best_config = args
    total_queries_made = 0


    time_start = time.time()

    for i in range(num_iterations):
        stu_lr = round(np.random.uniform(student_lr_range[0], student_lr_range[1]), 5)
        gen_lr = round(np.random.uniform(generator_lr_range[0], generator_lr_range[1]), 6)

        steps = [round(np.random.uniform(steps_range[0][0], steps_range[1][0]), 3),
                 round(np.random.uniform(steps_range[0][1], steps_range[1][1]), 3),
                 round(np.random.uniform(steps_range[0][2], steps_range[1][2]), 3)]

        scale = round(np.random.uniform(scale_range[0], scale_range[1]), 3)

        set_args(args, target_accuracy=target_accuracy, lr_tune_g=gen_lr, lr_tune_s=stu_lr,
                 steps=steps, scale=scale, batch_size=batch_size)
        args_set, learning_rate, query_per_epoch, accumulative_queries, total_queries, num_epochs, \
        time_taken = run_dfme_no_attack(args)

        total_queries_made += total_queries
        final_student_accuracy = copy.copy(learning_rate)[-1]

        if num_epochs == best_num_epochs:
            if final_student_accuracy > best_accuracy:
                best_num_epochs = num_epochs
                best_config = copy.copy(args_set)
                best_accuracy = final_student_accuracy

        elif num_epochs < best_num_epochs:
            best_num_epochs = num_epochs
            best_config = copy.copy(args_set)
            best_accuracy = final_student_accuracy

        save_grid_data(dataset=dataset, lr_tune_s=stu_lr, lr_tune_g=gen_lr, z_dim=128,
                       num_epochs=num_epochs, steps=steps, scale=scale, time_taken=time_taken,
                       student_accuracy=final_student_accuracy, file_to_write_to=file_to_write_to)

        print("completed step", i + 1, " out of ", num_iterations)

    time_end = time.time()
    total_time_taken = time_end - time_start
    return best_config, best_num_epochs, best_accuracy, total_queries_made, student_lr_range, generator_lr_range, steps_range, scale_range, total_time_taken, final_student_accuracy


if __name__ == '__main__':
    file_to_write_to = 'random_fashion_MNIST_momentum_0.5_1.txt'
    num_iterations = 25

    print("Starting random search")
    best_config, best_num_epochs, best_accuracy, total_queries_made, grid_student_lr, grid_generator_lr, grid_steps,\
        grid_scale, total_time_taken, final_student_acc = specified_random_search(file_to_write_to, num_iterations=num_iterations)

    save_best_grid_data(grid_student_lr=grid_student_lr, grid_generator_lr=grid_generator_lr, grid_setps=grid_steps,
                        grid_scale=grid_scale, dataset=best_config.dataset, lr_tune_s=best_config.lr_tune_s,
                        lr_tune_g=best_config.lr_tune_g, z_dim=best_config.z_dim, num_epochs=best_num_epochs,
                        steps=best_config.steps, scale=best_config.scale, time_taken=total_time_taken,
                        final_student_acc=best_accuracy, file_to_write_to=file_to_write_to, total_queries_made=total_queries_made)
