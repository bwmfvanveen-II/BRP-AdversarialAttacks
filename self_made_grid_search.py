import copy
import time

import torch.cuda

from run_dfme_exp import get_args, set_args, run_dfme_no_attack


def step_multiplier(step_array, multiplier):
    copy_of_array = copy.copy(step_array)
    for i in range(0, len(step_array)):
        copy_of_array[i] = step_array[i] * multiplier

    return copy_of_array


def specified_grid_search(file_to_save_to="default_grid_search_file_to_save.txt"):
    dataset = "mnist"
    # dataset = "fashion_mnist"

    args = get_args(dataset=dataset, cuda=torch.cuda.is_available())

    target_accuracy = 45

    grid_student_lr = [0.001, 0.006, 0.0085]
    grid_generator_lr = [0.0001, 0.0004]
    grid_steps = [[0.1, 0.5, 0.9], [0.2, 0.6, 0.933]]
    grid_scale = [0.3]

    best_num_epochs = 51
    best_accuracy = 0
    best_config = args
    total_queries_made = 0

    time_start = time.time()

    position_count = 1
    for stu_lr in grid_student_lr:
        for gen_lr in grid_generator_lr:
            for steps in grid_steps:
                for scale in grid_scale:
                    set_args(args, target_accuracy=target_accuracy, lr_tune_g=gen_lr, lr_tune_s=stu_lr,
                             steps=steps, scale=scale, dataset=dataset)
                    args_set, learning_rate, query_per_epoch, accumulative_queries, total_queries, num_epochs, \
                    time_taken = run_dfme_no_attack(args)

                    total_queries_made += total_queries
                    final_student_accuracy = learning_rate[-1]

                    if num_epochs == best_num_epochs:
                        if final_student_accuracy > best_accuracy:
                            best_num_epochs = num_epochs
                            best_config = copy.copy(args_set)
                            best_accuracy = copy.copy(final_student_accuracy)

                    elif num_epochs < best_num_epochs:
                        best_num_epochs = num_epochs
                        best_config = copy.copy(args_set)
                        best_accuracy = copy.copy(final_student_accuracy)

                    save_grid_data(dataset=dataset, lr_tune_s=stu_lr, lr_tune_g=gen_lr, z_dim=128,
                                   num_epochs=num_epochs, steps=steps, scale=scale, time_taken=time_taken,
                                   student_accuracy=final_student_accuracy, file_to_write_to=file_to_save_to)

                    print("completed step", position_count,
                          " out of ", len(grid_steps) * len(grid_scale) * len(grid_student_lr) * len(grid_generator_lr))

                    position_count += 1

    time_end = time.time()
    total_time_taken = time_end - time_start
    return best_config, best_num_epochs, best_accuracy, total_queries_made, grid_student_lr, grid_generator_lr, grid_steps, grid_scale, total_time_taken, final_student_accuracy


def save_grid_data(dataset, lr_tune_s, lr_tune_g, z_dim, num_epochs, steps, scale, time_taken,
                   student_accuracy, file_to_write_to):
    important_info = "Dataset: " + dataset + "\nLearning rate of S: " + str(lr_tune_s) + \
                     "\nLearning rate of G: " + str(lr_tune_g) + "\nz_dim: " + str(z_dim) + "\nStep timings: " \
                     + str(steps) + ", and scale: " + str(scale) + "\nFinal student accuracy: " + str(student_accuracy)\
                     + "%\nTotal time taken to run: " + str(time_taken) + \
                     " seconds \n\nNumber of epochs taken to achieve this result: " + str(num_epochs)

    with open(file_to_write_to, 'a') as outputfile:
        # outputfile.write(json.dumps(vars(args.)))
        outputfile.write(important_info)
        outputfile.write("\n--------------------------------------------------------------------------"
                         "---------------------------------------------\n")


def save_best_grid_data(grid_student_lr, grid_generator_lr, grid_setps, grid_scale, dataset, lr_tune_s, lr_tune_g,
                        z_dim, num_epochs, steps, scale, time_taken, final_student_acc, total_queries_made, file_to_write_to):
    starting_grid_set = "Grid of student learning rates: " + str(grid_student_lr) + \
                        "\nGrid of generator learning rates: " + str(grid_generator_lr) + "\nGrid of step timings: " + \
                        str(grid_setps) + "\nGrid of scales used: " + str(grid_scale)

    important_info = "Dataset: " + dataset + "\nLearning rate of S: " + str(lr_tune_s) + \
                     "\nLearning rate of G: " + str(lr_tune_g) + "\nz_dim: " + str(z_dim) + "\nStep timings: " \
                     + str(steps) + ", and scale: " + str(scale) + "\nFinal student accuracy: " + str(final_student_acc) \
                     + "%\nTotal time taken to run: " + str(time_taken) + \
                     " seconds \nTotal number of queries made: " + str(total_queries_made) + \
                     "\n\nNumber of epochs taken to achieve this result: " + str(num_epochs)

    with open(file_to_write_to, 'a') as outputfile:
        # outputfile.write(json.dumps(vars(args.)))
        outputfile.write("\n--------------------------------------------------------------------------"
                         "---------------------------------------------\n")
        outputfile.write("This is the set of optimal data for the grid of results:\n")
        outputfile.write(starting_grid_set)
        outputfile.write("\nThe optimal set of hyperparameters were: \n")
        outputfile.write(important_info)
        outputfile.write("\n--------------------------------------------------------------------------"
                         "---------------------------------------------\n")
        outputfile.write("\n--------------------------------------------------------------------------"
                         "---------------------------------------------\n")


if __name__ == '__main__':
    file_to_save_to = 'grid_search_fashion_MNIST_2.txt'

    print("Starting grid search")
    best_config, best_num_epochs, best_accuracy, total_queries_made, grid_student_lr, grid_generator_lr, grid_steps,\
        grid_scale, total_time_taken, final_student_acc = \
        specified_grid_search(file_to_save_to=file_to_save_to)

    save_best_grid_data(grid_student_lr=grid_student_lr, grid_generator_lr=grid_generator_lr, grid_setps=grid_steps,
                        grid_scale=grid_scale, dataset=best_config.dataset, lr_tune_s=best_config.lr_tune_s,
                        lr_tune_g=best_config.lr_tune_g, z_dim=best_config.z_dim, num_epochs=best_num_epochs,
                        steps=best_config.steps, scale=best_config.scale, time_taken=total_time_taken,
                        final_student_acc=best_accuracy, total_queries_made=total_queries_made,
                        file_to_write_to=file_to_save_to)
