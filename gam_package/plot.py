from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from math import pi

def parallelPlot(dfp):
    parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
    plt.show()

def radarPlot(dfp, medoidsList):
    # create background
    col = dfp.pop('medoid')
    dfp.insert(0, 'medoid', col)
    categories = list(dfp)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
    plt.ylim(0, .3)
    # add plots
    for i in range(len(medoidsList)):
        medoidname = medoidsList[i].index[0]
        values = medoidsList[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
        ax.fill(angles, values, 'b', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def facetedRadarPlot(dfp, medoidsList):
    # creates a radar plot for the given medoid
    def make_spider(row, title, color):
        col = dfp.pop('medoid')
        dfp.insert(0, 'medoid', col)
        categories = list(dfp)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(2, 2, row + 1, polar=True, )
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        ax.set_rlabel_position(0)
        plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
        plt.ylim(0, .3)
        medoidname = medoidsList[row].index[0]
        values = medoidsList[row].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title(title, size=11, color=color, y=1.1)

    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    my_palette = get_cmap("Set2", len(medoidsList))

    #for each medoid call make_spider to create a radar plot
    for row in range(len(medoidsList)):
        make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
    plt.show()