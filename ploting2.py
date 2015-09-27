import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bar_chart(N, vertical, lateral, mixed):
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, vertical, width, color='r', edgecolor='none')
    rects2 = ax.bar(ind + width, lateral, width, color='b', edgecolor='none')
    rects3 = ax.bar(ind + 2 * width, mixed, width, color='g', edgecolor='none')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean of absolute value of the weights')
    ax.set_title('Mean of absolute value of the weights')

    ax.legend((rects1[0], rects2[0], rects3[0]),
              ('Vertical', 'Laterel', 'Multiplication'))

    plt.savefig('bar_chart.png')

N = 200
vertical = list(np.random.randn(N, 1))
lateral = list(np.random.randn(N, 1))
mixed = list(np.random.randn(N, 1))
bar_chart(N, vertical, lateral, mixed)
