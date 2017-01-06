import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

def read_results(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    time_ = []; mem_ = []
    for counter, line in enumerate(lines):
        words = line.split()
        time_.append(float(words[0]))
        mem_.append(float(words[5]))

    return np.array(time_), np.array(mem_) 


def read_binary_results(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    size_ = []; flop_ = []
    for counter, line in enumerate(lines):
        words = line.split()
        word = words[0]
        if 'K' in  word:
            ss = float(word[:-1])*1024.
            size_.append(ss)
        elif 'M' in word:
            ss = float(word[:-1])*1024.*1024.
            size_.append(ss)

    for counter, line in enumerate(lines):
        words = line.split()
        word = words[0]
        if 'K' in  word or 'M' in word:
            ff = int(lines[counter+1])
            flop_.append(ff)

    return np.array(flop_), np.array(size_)


def get_flop_reduction(flop_dp,flop_nodp):

    assert flop_dp.shape == flop_nodp.shape
    flop_dp *= 2
    flop_nodp[:7] *= 3
    flop_nodp[7:14] *= 4
    flop_nodp[14:] *= 5
    return flop_nodp - flop_dp

def get_interp(x,y):
    xvals = np.linspace(x.min(),x.max(),10)
    f = interp1d(x, y, kind="linear")
    yvals = f(xvals)
    return xvals, yvals



def compilation_benchmark_tabulate():
    """Tabulate results for latex
    """

    mm = 7

    flop_nodp_gcc, size_nodp_gcc = read_binary_results("binary_results_nodp_gcc")
    flop_dp_gcc, size_dp_gcc     = read_binary_results("binary_results_dp_gcc")
    _, size_nodp_clang           = read_binary_results("binary_results_nodp_clang")
    _, size_dp_clang             = read_binary_results("binary_results_dp_clang")
    _, size_nodp_icc             = read_binary_results("binary_results_nodp_icc")
    _, size_dp_icc               = read_binary_results("binary_results_dp_gcc")

    reduced_flop = get_flop_reduction(flop_dp_gcc,flop_nodp_gcc).reshape(3,mm).T.copy()

    time_nodp_gcc, mem_nodp_gcc     = read_results("compilation_results_nodp_gcc")
    time_dp_gcc, mem_dp_gcc         = read_results("compilation_results_dp_gcc")
    time_nodp_clang, mem_nodp_clang = read_results("compilation_results_nodp_clang")
    time_dp_clang, mem_dp_clang     = read_results("compilation_results_dp_clang")
    time_nodp_icc, mem_nodp_icc     = read_results("compilation_results_nodp_icc")
    time_dp_icc, mem_dp_icc         = read_results("compilation_results_dp_icc")

    time_nodp_gcc   = time_nodp_gcc.reshape(3,mm).T.copy()
    time_nodp_clang = time_nodp_gcc.reshape(3,mm).T.copy()
    time_nodp_icc   = time_nodp_icc.reshape(3,mm).T.copy()
    time_dp_gcc     = time_dp_gcc.reshape(3,mm).T.copy()
    time_dp_clang   = time_dp_gcc.reshape(3,mm).T.copy()
    time_dp_icc     = time_dp_icc.reshape(3,mm).T.copy()

    mem_nodp_gcc   = mem_nodp_gcc.reshape(3,mm).T.copy()
    mem_nodp_clang = mem_nodp_gcc.reshape(3,mm).T.copy()
    mem_nodp_icc   = mem_nodp_icc.reshape(3,mm).T.copy()
    mem_dp_gcc     = mem_dp_gcc.reshape(3,mm).T.copy()
    mem_dp_clang   = mem_dp_gcc.reshape(3,mm).T.copy()
    mem_dp_icc     = mem_dp_icc.reshape(3,mm).T.copy()

    size_nodp_gcc   = size_nodp_gcc.reshape(3,mm).T.copy()
    size_nodp_gcc   = size_nodp_gcc.reshape(3,mm).T.copy()
    size_nodp_icc   = size_nodp_icc.reshape(3,mm).T.copy()
    size_dp_gcc     = size_dp_gcc.reshape(3,mm).T.copy()
    size_dp_clang   = size_dp_gcc.reshape(3,mm).T.copy()
    size_dp_icc     = size_dp_icc.reshape(3,mm).T.copy()


    for j in range(3):
        for i in range(mm):
            c1 = np.round(time_dp_gcc[i,j]/time_nodp_gcc[i,0],3)
            c2 = np.round(time_dp_clang[i,j]/time_nodp_clang[i,0],3)
            c3 = np.round(time_dp_icc[i,j]/time_nodp_icc[i,0],3)

            c4 = np.round(mem_dp_gcc[i,j]/mem_nodp_gcc[i,0],3)
            c5 = np.round(mem_dp_clang[i,j]/mem_nodp_clang[i,0],3)
            c6 = np.round(mem_dp_icc[i,j]/mem_nodp_icc[i,0],3)


            print reduced_flop[i,j], "&", c1, "&", c2, "&", c3, "&", c4, "&", c5, "&", c6, r"\\\hline"
        print 


    # nn = 2
    # plt.semilogx(reduced_flop[:,nn],time_dp_gcc[:,nn]/time_nodp_gcc[:,nn], linewidth=linewidth)
    # plt.semilogx(reduced_flop[:,nn],time_dp_clang[:,nn]/time_nodp_clang[:,nn], linewidth=linewidth)
    # plt.semilogx(reduced_flop[:,nn],time_dp_icc[:,nn]/time_nodp_icc[:,nn], linewidth=linewidth)
    # plt.grid('on')
    # plt.show()
    # exit()



if __name__ == "__main__":

    compilation_benchmark_tabulate()

