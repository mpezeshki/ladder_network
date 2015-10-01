import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 14,
          'font.family': 'lmodern',
          'text.latex.unicode': True}
plt.rcParams.update(params)


def parse_log(path, to_be_plotted):
    results = {}
    log = open(path, 'r').readlines()
    for line in log:
        colon_index = line.find(":")
        enter_index = line.find("\n")
        if colon_index != -1:
            key = line[:colon_index]
            value = line[colon_index + 1: enter_index]
            if key in to_be_plotted:
                values = results.get(key)
                if values is None:
                    results[key] = [value]
                else:
                    results[key] = results[key] + [value]
    return results


def pimp(path=None, xaxis='Epochs', yaxis='Cross Entropy', title=None):
    plt.legend(fontsize=14)
    plt.xlabel(r'\textbf{' + xaxis + '}')
    plt.ylabel(r'\textbf{' + yaxis + '}')
    plt.grid()
    plt.title(r'\textbf{' + title + '}')
    plt.ylim([0, 0.5])
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def plot(x, y, xlabel='train', ylabel='dev', color='b',
         x_steps=None, y_steps=None):
    if x_steps is None:
        x_steps = range(len(x))
    if y_steps is None:
        y_steps = range(len(y))
    plt.plot(x_steps, x, ls=':', c=color, lw=2, label=xlabel)
    plt.plot(y_steps, y, c=color, lw=2, label=ylabel)


def best(path, what='valid_Error_rate'):
    res = parse_log(path, [what])
    return min(res[what])


to_be_plotted = ['train_CE_clean', 'valid_CE_clean']
# to_be_plotted = ['train_Total_cost', 'valid_Total_cost']
yaxis = 'Cross Entropy'
# yaxis = 'Total cost'
titles = ['train ladder standard', 'valid ladder standard', 'train no bn', 'valid no bn']
main_title = 'no batch-norm'

file_1 = 'mnist_best/log.txt'
file_2 = 'mnist_2015_10_01_at_19_00/log.txt'


path = '/u/pezeshki/ladder_network/results/'
log1 = path + file_1
print best(log1)
results = parse_log(log1, to_be_plotted)
plt.figure()
plot(results[to_be_plotted[0]], results[to_be_plotted[1]],
     titles[0], titles[1], 'b')


log2 = path + file_2
print best(log2)
results = parse_log(log2, to_be_plotted)
plot(results[to_be_plotted[0]], results[to_be_plotted[1]],
     titles[2], titles[3], 'r')


pimp(path=None, yaxis=yaxis, title=main_title)
plt.savefig(path + 'plot.png')
