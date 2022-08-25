import sys

from plot.utils import plot_round_acc

accuracy_min = [74.78, 81.2, 84.34, 84.3, 84.38, 83.56, 83.66, 84.0, 83.74, 83.9, 84.04, 84.66, 85.18, 84.06, 85.16, 84.62, 83.46, 83.12, 84.24, 83.74, 82.26, 83.26, 81.42, 84.06, 82.42, 84.14, 82.58, 82.06, 81.9, 84.58, 83.14, 83.36, 82.52, 84.9, 84.44, 84.74, 84.02, 81.14, 82.84, 83.88, 82.32, 83.7, 84.42, 83.3, 83.04, 82.52, 82.8, 84.72, 83.88, 81.68]
accuracy_max = [84.46, 86.16, 86.14, 86.48, 85.94, 87.04, 86.42, 86.68, 87.9, 86.76, 86.76, 88.46, 86.82, 86.58, 87.84, 86.38, 87.16, 86.42, 86.9, 87.22, 86.14, 87.08, 87.22, 86.32, 87.58, 86.5, 86.52, 85.7, 87.34, 86.52, 86.84, 86.14, 86.72, 86.68, 86.08, 86.34, 86.38, 86.04, 86.92, 86.46, 86.98, 86.2, 87.12, 86.66, 85.72, 87.14, 86.26, 87.04, 85.84, 86.38]
accuracy_mean = [81.36, 84.28, 85.23, 85.26, 85.28, 85.21, 84.94, 85.42, 85.98, 85.23, 85.33, 85.75, 85.7, 85.36, 86.02, 85.34, 84.85, 85.44, 85.45, 85.64, 84.78, 85.36, 85.01, 85.14, 85.66, 85.22, 85.12, 84.19, 85.22, 85.47, 85.35, 84.81, 84.85, 85.62, 85.27, 85.38, 85.26, 83.58, 85.09, 84.93, 84.64, 85.2, 85.32, 85.26, 84.89, 84.66, 84.98, 85.68, 84.86, 84.98]
datasize_min = [78.52, 82.92, 83.46, 84.68, 84.28, 84.94, 83.42, 84.74, 83.46, 84.02, 83.92, 83.26, 83.86, 85.78, 83.62, 84.88, 83.92, 85.04, 84.54, 85.52, 85.56, 84.52, 83.28, 83.36, 84.48, 84.76, 84.3, 84.4, 84.24, 83.78, 84.52, 83.34, 84.34, 84.56, 83.72, 83.96, 85.22, 82.54, 84.94, 84.92, 83.92, 84.56, 84.8, 84.36, 85.66, 85.0, 83.24, 85.28, 83.22, 83.06]
datasize_max = [87.04, 86.72, 88.4, 86.88, 88.6, 88.42, 87.74, 88.38, 87.22, 86.86, 87.82, 87.76, 86.62, 87.56, 87.26, 87.4, 87.8, 87.82, 87.1, 88.08, 88.14, 87.22, 87.98, 87.3, 88.08, 87.18, 87.14, 87.62, 87.36, 87.68, 86.84, 87.2, 86.82, 87.54, 87.3, 87.68, 87.86, 87.84, 87.5, 87.1, 87.3, 87.52, 87.16, 87.2, 87.32, 87.08, 88.1, 87.6, 87.18, 87.72]
datasize_mean = [83.02, 84.83, 85.42, 85.73, 86.14, 85.96, 85.96, 85.9, 85.21, 85.45, 85.46, 85.83, 85.63, 86.32, 85.41, 85.77, 85.48, 86.3, 85.94, 86.38, 86.23, 86.2, 86.24, 85.9, 86.6, 86.05, 85.69, 85.77, 85.64, 86.26, 85.59, 86.02, 85.73, 86.32, 86.02, 86.01, 86.09, 85.73, 85.89, 85.77, 85.96, 86.05, 86.13, 85.8, 86.61, 86.4, 85.62, 86.36, 85.86, 85.46]
entropy_min = [74.06, 83.94, 84.68, 84.8, 84.06, 84.92, 82.36, 82.44, 85.6, 84.82, 85.28, 85.2, 84.08, 83.78, 83.76, 84.36, 84.14, 83.22, 84.48, 83.64, 84.36, 84.34, 84.26, 83.5, 82.14, 82.76, 83.7, 84.8, 84.0, 84.06, 84.18, 83.08, 84.48, 82.7, 83.96, 84.18, 83.32, 84.4, 81.2, 81.06, 84.02, 84.5, 84.86, 83.46, 83.6, 84.6, 84.52, 84.64, 85.7, 85.32]
entropy_max = [84.88, 86.02, 87.2, 87.48, 87.4, 88.2, 87.5, 87.24, 87.16, 87.74, 88.16, 87.24, 87.5, 87.56, 87.3, 88.1, 88.02, 87.12, 86.8, 88.14, 86.22, 87.28, 86.24, 87.72, 86.58, 85.76, 86.38, 88.36, 85.76, 85.74, 86.94, 86.6, 85.68, 87.44, 85.88, 87.76, 88.12, 88.32, 87.16, 87.58, 87.02, 87.14, 87.82, 87.0, 86.62, 86.82, 86.22, 87.56, 86.62, 87.2]
entropy_mean = [81.54, 85.23, 85.72, 86.43, 85.91, 86.41, 85.32, 86.05, 86.31, 86.1, 86.76, 86.19, 85.39, 85.77, 85.7, 86.09, 85.99, 85.66, 85.57, 85.56, 85.44, 85.76, 85.63, 85.53, 85.33, 84.57, 85.2, 86.06, 85.14, 85.25, 85.6, 85.3, 85.1, 85.18, 85.1, 85.35, 85.53, 85.79, 85.35, 85.34, 85.58, 85.39, 85.66, 85.61, 85.2, 85.46, 85.41, 85.91, 86.33, 86.13]
gradiv_max_min = [79.34, 85.02, 84.6, 84.6, 83.0, 84.64, 85.7, 85.14, 84.26, 84.08, 85.62, 83.86, 84.02, 85.78, 84.98, 84.82, 84.84, 84.92, 84.3, 84.82, 84.4, 84.72, 84.86, 84.24, 84.52, 84.92, 84.26, 83.6, 84.34, 84.94, 84.72, 85.28, 84.16, 84.64, 85.1, 83.16, 84.46, 84.92, 84.78, 83.22, 85.1, 84.44, 84.64, 83.64, 82.48, 84.28, 83.58, 83.82, 84.36, 83.5]
gradiv_max_max = [84.44, 86.82, 87.76, 87.58, 87.5, 88.22, 87.34, 88.94, 87.34, 87.92, 87.52, 87.64, 87.7, 87.94, 86.98, 86.92, 87.86, 87.62, 88.06, 86.7, 88.02, 86.52, 88.02, 87.84, 87.8, 87.7, 86.12, 87.78, 86.06, 87.36, 87.96, 87.02, 86.76, 86.22, 86.34, 87.36, 87.32, 86.86, 87.2, 86.28, 87.26, 86.94, 87.48, 87.14, 86.86, 86.94, 88.24, 87.02, 86.6, 87.48]
gradiv_max_mean = [82.56, 86.04, 86.21, 86.23, 85.93, 86.12, 86.7, 86.27, 85.88, 85.95, 86.38, 86.22, 85.83, 86.87, 85.92, 85.83, 86.24, 86.41, 86.05, 85.92, 86.04, 85.66, 86.2, 85.93, 85.94, 86.42, 85.43, 86.2, 85.38, 86.49, 85.94, 85.92, 85.26, 85.43, 85.7, 85.28, 85.72, 85.85, 85.95, 85.14, 86.05, 85.44, 85.69, 85.41, 85.05, 85.5, 85.82, 85.35, 85.42, 85.5]
gradiv_min_min = [75.24, 83.04, 84.88, 84.72, 82.78, 83.32, 84.28, 85.28, 84.3, 84.14, 85.1, 85.4, 82.6, 84.9, 84.62, 85.0, 84.86, 84.9, 84.0, 85.2, 84.22, 85.32, 84.76, 84.3, 83.38, 83.24, 84.42, 84.02, 84.84, 84.46, 83.86, 84.56, 83.9, 84.64, 84.26, 83.84, 84.5, 84.38, 84.28, 84.6, 83.96, 84.0, 84.4, 84.24, 82.82, 83.64, 85.38, 83.76, 83.02, 84.86]
gradiv_min_max = [84.26, 85.78, 86.66, 86.64, 86.98, 86.62, 86.66, 86.86, 86.3, 86.38, 86.86, 86.5, 87.22, 86.36, 86.6, 87.2, 87.1, 86.52, 86.48, 86.98, 86.3, 86.44, 86.88, 87.12, 87.4, 85.7, 87.4, 87.14, 87.18, 87.58, 86.64, 86.5, 87.1, 87.12, 86.34, 86.78, 86.62, 87.08, 87.6, 87.2, 87.08, 86.56, 87.04, 86.78, 86.68, 86.62, 86.48, 85.44, 87.3, 86.38]
gradiv_min_mean = [81.1, 84.47, 85.49, 85.76, 85.33, 85.09, 85.62, 86.08, 85.34, 85.39, 85.8, 86.14, 85.48, 85.65, 85.68, 86.03, 85.97, 85.64, 85.28, 86.34, 85.39, 85.64, 85.89, 85.53, 85.61, 84.8, 85.84, 85.77, 85.81, 85.79, 85.34, 85.58, 84.95, 85.89, 85.33, 85.19, 85.46, 86.0, 85.88, 85.96, 85.55, 85.34, 85.73, 85.97, 85.22, 85.33, 85.9, 84.87, 85.53, 85.73]
loss_min = [53.76, 83.54, 85.44, 85.2, 84.36, 84.76, 84.98, 85.04, 84.62, 85.06, 85.14, 84.84, 85.5, 84.4, 85.4, 83.9, 84.88, 84.9, 84.94, 85.38, 83.86, 84.68, 84.56, 85.72, 85.3, 84.1, 84.96, 85.12, 84.72, 85.46, 83.2, 84.96, 83.5, 85.18, 84.12, 85.34, 85.08, 85.3, 84.1, 84.44, 84.64, 84.3, 85.84, 84.58, 84.06, 83.58, 83.28, 84.06, 84.84, 85.04]
loss_max = [85.72, 85.94, 86.7, 87.54, 88.1, 87.66, 87.0, 87.36, 87.96, 87.64, 87.0, 87.16, 86.78, 86.86, 86.9, 87.7, 86.96, 87.1, 87.66, 87.74, 87.46, 87.4, 86.6, 86.68, 87.26, 87.78, 86.66, 86.98, 86.66, 86.98, 86.62, 86.56, 86.0, 86.92, 86.62, 86.64, 87.32, 87.42, 87.64, 86.82, 86.42, 86.5, 86.88, 87.08, 87.34, 87.04, 86.4, 85.9, 87.62, 86.62]
loss_mean = [77.98, 85.29, 86.19, 85.88, 86.48, 86.61, 85.93, 86.25, 86.21, 86.4, 86.0, 86.04, 86.15, 85.6, 86.25, 85.83, 86.01, 86.28, 86.16, 86.27, 85.93, 85.75, 85.77, 86.11, 86.15, 85.57, 85.98, 85.98, 85.69, 86.41, 84.93, 85.75, 85.22, 85.97, 85.75, 85.75, 86.14, 86.23, 85.71, 85.58, 85.67, 85.72, 86.37, 85.95, 85.78, 85.18, 85.23, 85.3, 85.78, 86.26]
random_min = [77.4, 82.68, 83.86, 85.02, 84.42, 84.96, 84.1, 85.08, 84.7, 82.2, 84.02, 84.7, 85.16, 85.38, 84.04, 85.26, 84.48, 85.04, 83.76, 83.56, 84.28, 84.7, 84.44, 83.04, 84.2, 83.84, 85.3, 84.5, 84.72, 84.58, 85.54, 84.04, 84.28, 84.06, 84.54, 83.58, 84.52, 83.96, 82.64, 83.24, 84.44, 83.08, 85.72, 83.58, 83.5, 83.66, 82.9, 83.74, 83.88, 83.64]
random_max = [85.16, 86.74, 87.76, 88.08, 86.9, 88.04, 87.58, 87.26, 87.56, 86.92, 87.74, 86.86, 86.46, 86.5, 86.56, 87.0, 86.32, 87.46, 87.1, 87.5, 86.82, 86.98, 87.24, 87.22, 87.58, 86.58, 86.66, 87.0, 86.66, 86.76, 87.22, 87.02, 87.02, 86.56, 86.86, 87.24, 87.3, 86.8, 86.84, 87.48, 86.8, 86.58, 86.9, 86.52, 87.32, 86.82, 87.2, 87.2, 86.66, 87.58]
random_mean = [82.03, 84.76, 86.12, 86.28, 85.77, 86.3, 86.36, 85.79, 85.84, 85.07, 85.68, 85.72, 85.77, 86.11, 85.52, 86.05, 85.77, 86.47, 85.47, 85.62, 85.43, 85.87, 85.7, 85.43, 85.75, 85.12, 85.73, 85.59, 85.58, 86.1, 86.26, 85.35, 85.48, 85.2, 85.19, 85.72, 85.64, 85.29, 85.07, 85.42, 85.3, 84.99, 86.2, 85.48, 85.42, 84.76, 84.9, 85.55, 85.16, 85.18]

data = {
    "accuracy_min": accuracy_min,
    "accuracy_max": accuracy_max,
    "accuracy_mean": accuracy_mean,
    "datasize_min": datasize_min,
    "datasize_max": datasize_max,
    "datasize_mean": datasize_mean,
    "entropy_min": entropy_min,
    "entropy_max": entropy_max,
    "entropy_mean": entropy_mean,
    "gradiv_max_min": gradiv_max_min,
    "gradiv_max_max": gradiv_max_max,
    "gradiv_max_mean": gradiv_max_mean,
    "gradiv_min_min": gradiv_min_min,
    "gradiv_min_max": gradiv_min_max,
    "gradiv_min_mean": gradiv_min_mean,
    "loss_min": loss_min,
    "loss_max": loss_max,
    "loss_mean": loss_mean,
    "random_min": random_min,
    "random_max": random_max,
    "random_mean": random_mean,
}

save_path = None
if len(sys.argv) == 3 and sys.argv[1] and sys.argv[1] == "save":
    save_path = sys.argv[2]

plot_round_acc("", data, legend_pos="in", save_path=save_path, plot_size="2", y_lim_bottom=80)
