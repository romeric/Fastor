import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'],'size':14})
rc('text', usetex=True)

def read_results():

    ms, ns, times_eigen, times_blaze, times_fastor = [], [], [], [], []
    with open("benchmark_results.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        sline = line.split(' ')
        if len(sline) == 4:
            times_eigen.append(float(sline[1]))
            times_blaze.append(float(sline[2]))
            times_fastor.append(float(sline[3]))
        elif len(sline) == 6 and "size" in sline[1]:
            ms.append(int(sline[4]))
            ns.append(int(sline[5]))

    return np.array(ms), np.array(ns), np.array(times_eigen), np.array(times_blaze), np.array(times_fastor)


def main():

    ms, ns, times_eigen, times_blaze, times_fastor = read_results()

    fig, ax = plt.subplots()
    index = np.arange(len(ms))
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, times_eigen/1e-6, bar_width,
        alpha=opacity,
        color='#C03B22',
        label='Eigen')

    rects2 = plt.bar(index + bar_width, times_blaze/1e-6, bar_width,
        alpha=opacity,
        color='#A48568',
        label='Blaze')

    rects3 = plt.bar(index + 2*bar_width, times_fastor/1e-6, bar_width,
        alpha=opacity,
        color='#E98604',
        label='Fastor')


    xticks = [str(dim[0]) + 'x' + str(dim[1]) for dim in zip(ms,ns)]
    plt.xlabel('(M,N)')
    plt.ylabel('Time ($\mu$sec)')
    plt.title("B = transpose(A)")
    plt.xticks(index, xticks, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    # plt.savefig('benchmark_transpose_single.png', format='png', dpi=300)
    # plt.savefig('benchmark_transpose_double.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()