import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'],'size':10})
rc('text', usetex=True)

def read_results():

    mnk, times = [], []
    with open("mkl_jit_benchmark_results_double_7.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        sline = line.split(' ')
        if len(sline) > 5 and len(sline) < 10:
            mnk.append([int(sline[5]),int(sline[6]),int(sline[7])])
        elif len(sline) > 10:
            if sline[5] == "s.":
                times.append(float(sline[4]))
            elif sline[5] == "ms.":
                times.append(float(sline[4])*1e-3)
            elif sline[5] == "ns.":
                times.append(float(sline[4])*1e-9)
            else:
                times.append(float(sline[4])*1e-6)

    mnk = np.array(mnk)
    times = np.array(times)
    times = times.reshape(int(times.shape[0]/5),5)

    return mnk, times


def main():

    mnk, times = read_results()
    mnk = mnk[:,[0,2,1]]
    ntests = times.shape[1]
    for i in range(1,ntests):
        times[:,i] /= times[:,0]
    times[:,0] = 1.

    fig, ax = plt.subplots()
    index = np.arange(times.shape[0])
    bar_width = 0.15
    opacity = 0.8

    rects3 = plt.bar(index, times[:,0], bar_width,
        alpha=opacity,
        color='#E98604',
        label='Fastor')

    rects2 = plt.bar(index + bar_width, times[:,1], bar_width,
        alpha=opacity,
        color='#A48568',
        label='Blaze Unpadded')

    rects3 = plt.bar(index + 2*bar_width, times[:,2], bar_width,
        alpha=opacity,
        color='#4D5C75',
        label='Blaze Padded')

    rects4 = plt.bar(index + 3*bar_width, times[:,3], bar_width,
        alpha=opacity,
        color='#D1655B',
        label='MKL DIRECT CALL SEQ JIT')

    rects5 = plt.bar(index + 4*bar_width, times[:,4], bar_width,
        alpha=opacity,
        color='#697F3F',
        label='MKL JIT API')


    xticks = [str(dim[0]) + 'x' + str(dim[1]) + 'x' + str(dim[2]) for dim in mnk]
    plt.xlabel('(M,N,K)')
    plt.ylabel('Normalised Runtime')
    plt.title('DGEMM [C = A*B]')
    plt.xticks(index, xticks, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    # plt.savefig('mkl_jit_benchmark_results_double_7_cloud.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()