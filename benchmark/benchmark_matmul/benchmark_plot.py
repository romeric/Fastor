import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'],'size':14})
rc('text', usetex=True)

def read_results():

    mnk, gflops, times = [], [], []
    # with open("benchmark_results_single_linux.txt", "r") as f:
    with open("benchmark_results_double_linux.txt", "r") as f:
    # with open("benchmark_results_single.txt", "r") as f:
    # with open("benchmark_results_double.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        sline = line.split(' ')
        if len(sline) > 5:
            mnk.append([int(sline[3]),int(sline[4]),int(sline[5])])
            gflops.append(float(sline[7]))
            times.append(float(sline[10]))

    mnk = np.array(mnk[:len(mnk) / 4])
    gflops = np.array(gflops)
    times = np.array(times)

    return mnk, gflops, times


def main():

    mnk, gflops, times = read_results()
    ntests = len(gflops) / 4

    gflops_eigen = gflops[:ntests]
    gflops_blaze = gflops[ntests:2*ntests]
    gflops_fastor = gflops[2*ntests:3*ntests]
    gflops_xsmm = gflops[3*ntests:4*ntests]

    times_eigen = times[:ntests]
    times_blaze = times[ntests:2*ntests]
    times_fastor = times[2*ntests:3*ntests]
    times_xsmm = times[3*ntests:4*ntests]

    fig, ax = plt.subplots()
    index = np.arange(ntests)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, gflops_eigen, bar_width,
        alpha=opacity,
        color='#C03B22',
        label='Eigen')

    rects2 = plt.bar(index + bar_width, gflops_blaze, bar_width,
        alpha=opacity,
        color='#A48568',
        label='Blaze')

    rects3 = plt.bar(index + 2*bar_width, gflops_fastor, bar_width,
        alpha=opacity,
        color='#E98604',
        label='Fastor')

    rects4 = plt.bar(index + 3*bar_width, gflops_xsmm, bar_width,
        alpha=opacity,
        color='#697F3F',
        label='LIBXSMM')


    xticks = [str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) for dim in mnk]
    plt.xlabel('(M,N,K)')
    plt.ylabel('GFlops/s')
    plt.title('DGEMM [C = A*B]')
    plt.xticks(index, xticks, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    # plt.savefig('benchmark_results_single.png', format='png', dpi=300)
    # plt.savefig('benchmark_results_double.png', format='png', dpi=300)
    # plt.savefig('benchmark_results_single_linux.png', format='png', dpi=300)
    plt.savefig('benchmark_results_double_linux.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()