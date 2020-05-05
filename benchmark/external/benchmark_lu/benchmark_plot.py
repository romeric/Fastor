import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'],'size':14})
rc('text', usetex=True)

def read_results():

    ms, ns, times_eigen, times_fastor = [], [], [], []
    with open("benchmark_results.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        sline = line.split(' ')
        if len(sline) == 3:
            times_eigen.append(float(sline[1]))
            times_fastor.append(float(sline[2]))
        elif len(sline) > 4 and "size" in sline[1]:
            ms.append(int(sline[4]))
            ns.append(int(sline[5]))

    return np.array(ms), np.array(ns), np.array(times_eigen), np.array(times_fastor)


def main():

    ms, ns, times_eigen, times_fastor = read_results()

    fig, ax = plt.subplots()
    index = np.arange(len(ms))
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, times_eigen/1e-6, bar_width,
        alpha=opacity,
        color='#C03B22',
        label='Eigen')

    rects3 = plt.bar(index + bar_width, times_fastor/1e-6, bar_width,
        alpha=opacity,
        color='#E98604',
        label='Fastor')


    xticks = [str(dim[0]) + 'x' + str(dim[1]) for dim in zip(ms,ns)]
    plt.xlabel('(M,M)')
    plt.ylabel('Time ($\mu$sec)')
    plt.title("[L, U, P] = lu(A)")
    plt.xticks(index, xticks, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    # plt.savefig('benchmark_lu_single.png', format='png', dpi=300)
    # plt.savefig('benchmark_lu_double.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()