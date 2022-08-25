import os

from utils import calculate_average_across_files


def extract_round_acc():
    base_path = "./output/result-v13/non_iid/cnn-fmnist"
    experiment_names = ["accuracy", "datasize", "entropy", "gradiv_max", "gradiv_min", "loss", "random"]
    # experiment_names = ["accuracy-0.3", "accuracy-0.5", "accuracy-0.7"]
    end_dirname = "output"

    for experiment_name in experiment_names:
        output_path = os.path.join(base_path, experiment_name, end_dirname)
        files_numbers_mean_2d_np = calculate_average_across_files(output_path)
        acc = [round(i, 2) for i in files_numbers_mean_2d_np[:, 5]]
        print(experiment_name, "=", acc)


def main():
    extract_round_acc()


if __name__ == "__main__":
    main()
