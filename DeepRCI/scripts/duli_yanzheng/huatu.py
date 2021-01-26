import matplotlib.pyplot as plt
import numpy as np

def huatu(aac1, aac2, aac3):
    n_groups = 5

    # means_men = (0.9829, 0.9881, 0.978, 0.9657)
    # std_men = (1, 1, 1, 1, 1)

    # means_women = (0.89, 0.87, 0.92, 0.79)
    # std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, aac1, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label="Mbah's method" )

    rects2 = ax.bar(index + bar_width, aac2, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label="Hou's method")

    rects3 = ax.bar(index + 2 * bar_width, aac3, bar_width,
                    alpha=opacity, color='black',
                    error_kw=error_config,
                    label="DeepRCI")

    plt.ylim([0, 0.15])
    ax.set_title("Amino acid composition")
    ax.set_xticks(index + bar_width / 2)
    plt.ylim([0.2, 1.2])
    ax.set_title("Comparison with other method")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Acc', 'Spec', 'Sens', 'F-score', 'Mcc'))
    ax.legend()
    fig.tight_layout()
    plt.show()

def main():
    acc2 = [0.8113, 0.8469, 0.7731, 0.7878, 0.6228]
    acc1 = [0.8008, 0.8176, 0.784, 0.7873, 0.602]
    acc3 = [0.9319, 0.9360, 0.9279, 0.9353, 0.8638]
    huatu(acc1, acc2, acc3)

if __name__ == '__main__':
    main()