import numpy as np


# calculate average on any number of data lists
def avg_calculator(data_list):
    data_2d_np = np.array(data_list)
    data_mean_np = data_2d_np.mean(axis=0)
    data_max_np = np.amax(data_2d_np, axis=0)
    data_min_np = np.amin(data_2d_np, axis=0)
    min_list = [round(i, 2) for i in data_min_np]
    max_list = [round(i, 2) for i in data_max_np]
    mean_list = [round(i, 2) for i in data_mean_np]
    return min_list, max_list, mean_list


def resolve_parallel_results(parallel_results):
    for i_scheme in range(len(parallel_results[0])):
        data_list = []
        for parallel_result in parallel_results:
            data_list.append(parallel_result[i_scheme])
        min_list, max_list, mean_list = avg_calculator(data_list)
        print("_min = {}".format(min_list))
        print("_max = {}".format(max_list))
        print("_mean = {}".format(mean_list))
        print("")


def main():
    result_data_01 = [

    ]

    result_data_02 = [

    ]

    result_data_03 = [

    ]

    result_data_04 = [

    ]

    result_data_05 = [

    ]

    parallel_results = [
        result_data_01,
        result_data_02,
        result_data_03,
        result_data_04,
        result_data_05,
    ]
    resolve_parallel_results(parallel_results)


if __name__ == "__main__":
    main()
