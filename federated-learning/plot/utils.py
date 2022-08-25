# -*- coding: UTF-8 -*-

# For ubuntu env error: findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans.
# ```bash
# sudo apt install msttcorefonts
# rm -rf ~/.cache/matplotlib
# ```
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from cycler import cycler
import pylab

# input latex symbols in matplotlib
# https://stackoverflow.com/questions/43741928/matplotlib-raw-latex-epsilon-only-yields-varepsilon
plt.rcParams["mathtext.fontset"] = "cm"


# Plot number in a row: "2", "3", "4"
# 2: Two plots in a row (the smallest fonts)
# 3: Three plots in a row
# 4: Four plots in a row (the biggest fonts)
def get_font_settings(size):
    if size == "2":
        font_size_dict = {"l": 21, "m": 18, "s": 16}
        fig_width = 8  # by default is 6.4 x 4.8
        fig_height = 4
    elif size == "3":
        font_size_dict = {"l": 25, "m": 21, "s": 19}
        fig_width = 8
        fig_height = 4
    else:
        font_size_dict = {"l": 25, "m": 25, "s": 20}
        # fig_width = 6.4
        # fig_height = 4.8
        fig_width = 7.4
        fig_height = 3.7

    xy_label_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["l"])
    title_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["m"])
    legend_font = font_manager.FontProperties(
        family='Times New Roman', weight='bold', style='normal', size=font_size_dict["s"])
    ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', size=font_size_dict["s"])
    cs_xy_label_font = {'fontproperties': xy_label_font}
    cs_title_font = {'fontproperties': title_font}
    cs_xy_ticks_font = {'fontproperties': ticks_font}
    font_factory = {
        'legend_font': legend_font,
        'cs_xy_label_font': cs_xy_label_font,
        'cs_title_font': cs_title_font,
        'cs_xy_ticks_font': cs_xy_ticks_font,
        'fig_width': fig_width,
        'fig_height': fig_height,
    }
    return font_factory


def get_color_settings():
    # color names: https://matplotlib.org/stable/gallery/color/named_colors.html
    # colors = plt.get_cmap('tab10').colors  # by default
    colors = ("tab:blue",) + plt.get_cmap('Set2').colors
    # colors = [plt.cm.Spectral(i / float(6)) for i in range(6)]
    # colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    return colors


def get_cycler_settings():
    my_cycler = cycler(color=get_color_settings())
    return my_cycler


def plot_legend_head(axes, legend_column, width, height, save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    figlegend = pylab.figure()
    figlegend.legend(axes.get_legend_handles_labels()[0], axes.get_legend_handles_labels()[1],
                     prop=font_settings.get("legend_font"), ncol=legend_column, loc='upper center')
    figlegend.tight_layout()
    figlegend.set_size_inches(width, height)
    if save_path:
        save_path = save_path[:-4] + "-legend.png"
        figlegend.savefig(save_path)
    else:
        figlegend.show()


# data: {"scheme01": scheme01_data, "scheme02": scheme02_data}
# legend: legend position. Values: "in", "out", or nothing
def plot_round_acc(title, data, legend_pos="", save_path=None, plot_size="3", y_lim_bottom=None):
    font_settings = get_font_settings(plot_size)
    cycler_settings = get_cycler_settings()
    x = range(1, 51)

    fig, axes = plt.subplots()
    axes.set_prop_cycle(cycler_settings)
    mycolors = cycler_settings.by_key()['color']

    axes.plot(x, data["accuracy_mean"], label="ACC", marker='o', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["accuracy_min"], data["accuracy_max"], linewidth=0, alpha=0.3, color=mycolors[0])
    axes.plot(x, data["datasize_mean"], label="DSZ", marker='D', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["datasize_min"], data["datasize_max"], linewidth=0, alpha=0.3, color=mycolors[1])
    axes.plot(x, data["entropy_mean"], label="ENT", marker='v', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["entropy_min"], data["entropy_max"], linewidth=0, alpha=0.3, color=mycolors[2])
    axes.plot(x, data["gradiv_max_mean"], label="G-MAX", marker='>', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["gradiv_max_min"], data["gradiv_max_max"], linewidth=0, alpha=0.3, color=mycolors[3])
    axes.plot(x, data["gradiv_min_mean"], label="G-MIN", marker='x', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["gradiv_min_min"], data["gradiv_min_max"], linewidth=0, alpha=0.3, color=mycolors[4])
    axes.plot(x, data["loss_mean"], label="LOSS", marker='|', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["loss_min"], data["loss_max"], linewidth=0, alpha=0.3, color=mycolors[5])
    axes.plot(x, data["random_mean"], label="RDM", marker='<', markevery=5, markersize=8, mfc='none')
    # axes.fill_between(x, data["random_min"], data["random_max"], linewidth=0, alpha=0.3, color=mycolors[6])

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    plt.ylim(bottom=y_lim_bottom)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 7, 12, 0.6, save_path, plot_size)


def plot_node_acc(title, data, legend_pos="", save_path=None, plot_size="3"):
    font_settings = get_font_settings(plot_size)
    cycler_settings = get_cycler_settings()
    x = range(1, 51)

    fig, axes = plt.subplots()
    axes.set_prop_cycle(cycler_settings)
    mycolors = cycler_settings.by_key()['color']

    axes.plot(x, data["accuracy_03_mean"], label="ACC-03", marker='o', markevery=5, markersize=8, mfc='none')
    axes.fill_between(x, data["accuracy_03_min"], data["accuracy_03_max"], linewidth=0, alpha=0.3, color=mycolors[0])
    axes.plot(x, data["accuracy_05_mean"], label="ACC-05", marker='D', markevery=5, markersize=8, mfc='none')
    axes.fill_between(x, data["accuracy_05_min"], data["accuracy_05_max"], linewidth=0, alpha=0.3, color=mycolors[1])
    axes.plot(x, data["accuracy_07_mean"], label="ACC-07", marker='v', markevery=5, markersize=8, mfc='none')
    axes.fill_between(x, data["accuracy_07_min"], data["accuracy_07_max"], linewidth=0, alpha=0.3, color=mycolors[2])

    axes.set_xlabel("Training Round", **font_settings.get("cs_xy_label_font"))
    axes.set_ylabel("Accuracy (%)", **font_settings.get("cs_xy_label_font"))

    plt.title(title, **font_settings.get("cs_title_font"))
    plt.xticks(**font_settings.get("cs_xy_ticks_font"))
    plt.yticks(**font_settings.get("cs_xy_ticks_font"))
    plt.tight_layout()
    # plt.ylim(bottom=70)
    if legend_pos == "in":
        plt.legend(prop=font_settings.get("legend_font"), loc='lower right').set_zorder(11)
    plt.grid()
    fig.set_size_inches(font_settings.get("fig_width"), font_settings.get("fig_height"))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    if legend_pos == "out":
        plot_legend_head(axes, 7, 12, 0.6, save_path, plot_size)

