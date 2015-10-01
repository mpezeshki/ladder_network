import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import get_streams
import theano
from blocks.graph import ComputationGraph


def bar_chart(params_dicts):
    vertical = []
    lateral = []
    mixed = []
    for i in range(7):
        name_vertical = 'g_' + str(i) + "_a3"
        name_lateral = 'g_' + str(i) + "_a2"
        name_mixed = 'g_' + str(i) + "_a4"
        vertical += [np.mean(np.abs(list(
            params_dicts[name_vertical].get_value())))]
        lateral += [np.mean(np.abs(list(
            params_dicts[name_lateral].get_value())))]
        mixed += [np.mean(np.abs(list(
            params_dicts[name_mixed].get_value())))]

    ind = np.arange(len(vertical))  # the x locations for the groups
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


def plot_representations(ladder, params_dicts):
    train_data_stream, valid_data_stream = get_streams(50000, 50000)
    data = train_data_stream.get_epoch_iterator().next()
    cg = ComputationGraph([ladder.costs.total])
    f = theano.function(
        [cg.inputs[0]],
        [ladder.wz[-1], ladder.wu[-1], ladder.wzu[-1], ladder.ests[-1]])

    wz, wu, wzu, est = f(data[0])

    plt.imshow(
        np.vstack(
            [np.swapaxes(
                data[0][8:18].reshape(10, 28, 28), 0, 1).reshape(28, 280),
             np.swapaxes(
                est[8:18].reshape(10, 28, 28), 0, 1).reshape(28, 280),
             np.swapaxes(
                wz[8:18].reshape(10, 28, 28), 0, 1).reshape(28, 280),
             np.swapaxes(
                wu[8:18].reshape(10, 28, 28), 0, 1).reshape(28, 280),
             np.swapaxes(
                wzu[8:18].reshape(10, 28, 28), 0, 1).reshape(28, 280),
             np.swapaxes(
                np.vstack([params_dicts['g_0_a1'].get_value() for i in
                           range(10)]).reshape(10, 28, 28), 0, 1).reshape(
                28, 280)]),
        cmap=plt.gray(), interpolation='nearest', vmin=0, vmax=1)
    plt.savefig('est.png')


def compute_noises(ladder):
    cg = ComputationGraph([ladder.costs.total])
    f_clean = theano.function([cg.inputs[0]], ladder.clean_zs)
    f_corr = theano.function([cg.inputs[0]], ladder.corr_zs)
    train_data_stream, valid_data_stream = get_streams(50000, 50000)
    data = train_data_stream.get_epoch_iterator().next()
    rs_clean = f_clean(data[0])
    rs_corr = f_corr(data[0])
    import ipdb; ipdb.set_trace()
    stds = [np.std(rs_clean[i] - rs_corr[i])
            for i in range(len(ladder.corr_zs))]
    print stds
    means = [np.mean(rs_clean[i] - rs_corr[i])
             for i in range(len(ladder.corr_zs))]
    print means
