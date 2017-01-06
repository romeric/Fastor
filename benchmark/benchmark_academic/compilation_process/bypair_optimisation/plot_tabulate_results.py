import os, sys
import numpy as np
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


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

    size_ = []
    for counter, line in enumerate(lines):
        words = line.split()
        word = words[0]
        if 'K' in  word:
            ss = float(word[:-1])*1024.
        elif 'M' in word:
            ss = float(word[:-1])*1024.*1024.
        size_.append(ss)


    return np.array(size_)


def read_runtime_results(filename):
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    runt_ = []
    for counter, line in enumerate(lines):
        words = line.split()
        runt_.append(float(words[0]))

    return np.array(runt_)



def compilation_benchmark_plots(quantity=0, save=False, figure_name=None):
    """
        quantity:               0 - Wall Time
                                1 - memory usage
                                2 - binary code size
                                3 - execution (run time)
    """

    if quantity==0 or quantity==1:

        time_gcc, mem_gcc = read_results("compilation_results_gcc")
        time_clang, mem_clang = read_results("compilation_results_clang")
        time_icc, mem_icc = read_results("compilation_results_icc")

    elif quantity==2:

        size_gcc = read_binary_results("binary_results_gcc")
        size_clang = read_binary_results("binary_results_clang")
        size_icc = read_binary_results("binary_results_icc")

    elif quantity==3:

        runtime_gcc = read_runtime_results("runtime_results_gcc")
        runtime_clang = read_runtime_results("runtime_results_clang")
        runtime_icc = read_runtime_results("runtime_results_icc")


    if quantity==0:
        quan_gcc = time_gcc
        quan_clang = time_clang
        quan_icc = time_icc
    elif quantity==1:
        mb = 1024.
        quan_gcc = mem_gcc/mb
        quan_clang = mem_clang/mb
        quan_icc = mem_icc/mb
    elif quantity==2:
        kb = 1024.
        quan_gcc = size_gcc/kb
        quan_clang = size_clang/kb
        quan_icc = size_icc/kb
    elif quantity==3:
        quan_gcc = runtime_gcc
        quan_clang = runtime_clang
        quan_icc = runtime_icc


    # MASK GCC RESULTS
    tmp = quan_gcc.copy()
    quan_gcc = np.zeros_like(quan_icc)
    quan_gcc[[0,1,2,3,4,5,6,7,8,9,10,12,13,15,16,18,19]] = tmp

    quans = np.zeros((quan_icc.shape[0],3))
    quans[:,0] = quan_gcc
    quans[:,1] = quan_clang
    quans[:,2] = quan_icc
    quans = np.fliplr(quans)

    orig_quans = quans.copy()
    xx = np.where(quans==0)[0]
    quans[quans==0] = 1.

    # TRANSFORM THE DATA TO FIT THE PLOTTING RANGE
    if quantity==0:
        quans = np.log(quans)/np.log(1.4)
    elif quantity==1:
        quans *= 2*1e-3 
        # quans = np.log(quans)/np.log(2.0)
    elif quantity==2:
        quans = np.log(quans)/np.log(2.0)
    elif quantity==3:
        quans *= 1.6*1e4 
        # quans *= 1e9
        # quans = np.log(quans)/np.log(3.1)
        quans[xx,2]=0.


    X = np.zeros((quan_icc.shape[0],3,1),dtype=np.int64)
    counter = 0
    for i in range(X.shape[0]):
        if counter in [3,7,11,15,19,23]:
            counter+=1
        X[i,:,:] = counter
        counter+=1
    xs = X[:,0,0]

    if quantity==3:
        max_mapper = orig_quans.max()*1000.
    else:
        max_mapper = orig_quans.max()

    from matplotlib.colors import ColorConverter
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab
    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(1100,800))

    # ACTUAL DATA SHOULD APPEAR IN THE COLORBAR
    h_dum = mlab.barchart(np.zeros_like(quans), mode="cube", colormap="autumn", vmin=0, vmax=max_mapper)
    # ACTUAL PLOT
    quans[quans==0] = np.NAN
    h = mlab.barchart(quans, mode="cube", colormap="autumn", mask_points=None)
    h.mlab_source.x = X
    h_dum.mlab_source.x = X

     # CHANGE LIGHTING OPTION
    h.actor.property.interpolation = 'phong'
    h.actor.property.specular = 0.1
    h.actor.property.specular_power = 5

    color_func = ColorConverter()
    rgba_lower = color_func.to_rgba_array(cm.viridis.colors)
    RGBA_higher = np.round(rgba_lower*255).astype(np.int64)
    h.module_manager.scalar_lut_manager.lut.table = RGBA_higher
    h_dum.module_manager.scalar_lut_manager.lut.table = RGBA_higher

    scale = (0.4,0.4,0.4)
    for i in range(quans.shape[0]):
        # if i in [0,3,6,9,12,15,18]:
        #     labeler = r"-DCONTRACT_OPT=0"
        # elif i in np.array([0,3,6,9,12,15,18])+1:
        #     labeler = r"-DCONTRACT_OPT=1"
        # elif i in np.array([0,3,6,9,12,15,18])+2:
        #     labeler = r"-DCONTRACT_OPT=2"
        if i in [0,3,6,9,12,15,18]:
            labeler = r"-DOPT=0"
        elif i in np.array([0,3,6,9,12,15,18])+1:
            labeler = r"-DOPT=1"
        elif i in np.array([0,3,6,9,12,15,18])+2:
            labeler = r"-DOPT=2"
        mlab.text3d(xs[i], -0.6, 0, labeler, scale=scale, orient_to_camera=False, orientation=(0,0,-90))
    
    mlab.text3d(-5, -.2, 0,  r"ICC 17.0.1",  scale=scale, orient_to_camera=False, orientation=(0,0,0))
    mlab.text3d(-5, 0.6, 0,  r"CLANG 3.9.0", scale=scale, orient_to_camera=False, orientation=(0,0,0))
    mlab.text3d(-5, 1.4, 0,  r"GCC 6.2.0",   scale=scale, orient_to_camera=False, orientation=(0,0,0))

    for i in range(7):
        mlab.text3d(4*i+.3, -4.4, 0, str(7-i)+r" Index", scale=scale, orient_to_camera=False, orientation=(0,0,0))


    # mlab.outline()
    if quantity==0:
        title = "Compilation Time (sec)"
        formatter = "%.2f"
    elif quantity==1:
        title = "Memory Usage (MB)     "
        formatter = "%.2f"
    elif quantity==2:
        title = "Binary Size (KB)      "
        formatter = "%.2f"
    elif quantity==3:
        title = "Execution Time (msec) "
        formatter = "%.4f"
    
    hc = mlab.colorbar(object=h_dum, title=title,orientation="vertical", nb_labels=6, label_fmt=formatter)
    hc.scalar_bar_representation.position = [0.01, 0.35] # x, y
    hc.scalar_bar_representation.position2 = [0.14, 0.5] # width, height
    hc.scalar_bar_representation.proportional_resize=True

    mlab.view(azimuth=255, elevation=45, distance=55, focalpoint=(12,8,0))

    if save:
        mlab.savefig(figure_name)
    mlab.show()



def compilation_benchmark_tabulate():
    """Tabulate optimisation levels
    """

    time_gcc, mem_gcc = read_results("compilation_results_gcc")
    time_clang, mem_clang = read_results("compilation_results_clang")
    time_icc, mem_icc = read_results("compilation_results_icc")

    size_gcc = read_binary_results("binary_results_gcc")
    size_clang = read_binary_results("binary_results_clang")
    size_icc = read_binary_results("binary_results_icc")

    runtime_gcc = read_runtime_results("runtime_results_gcc")
    runtime_clang = read_runtime_results("runtime_results_clang")
    runtime_icc = read_runtime_results("runtime_results_icc")




    # MASK GCC RESULTS
    def masker(quan_gcc,sizer):
        tmp = quan_gcc.copy()
        quan_gcc = np.zeros(sizer)
        quan_gcc[[0,1,2,3,4,5,6,7,8,9,10,12,13,15,16,18,19]] = tmp
        return quan_gcc

    sizer = time_icc.shape[0]
    time_gcc = masker(time_gcc,sizer)
    mem_gcc = masker(mem_gcc,sizer)
    size_gcc = masker(size_gcc,sizer)
    runtime_gcc = masker(runtime_gcc,sizer)

    rows = 21 // 3

    time_gcc = time_gcc.reshape(rows,3)
    time_clang = time_clang.reshape(rows,3)
    time_icc = time_icc.reshape(rows,3)

    mem_gcc = mem_gcc.reshape(rows,3)
    mem_clang = mem_clang.reshape(rows,3)
    mem_icc = mem_icc.reshape(rows,3)

    size_gcc = size_gcc.reshape(rows,3)
    size_clang = size_clang.reshape(rows,3)
    size_icc = size_icc.reshape(rows,3)

    runtime_gcc = runtime_gcc.reshape(rows,3)
    runtime_clang = runtime_clang.reshape(rows,3)
    runtime_icc = runtime_icc.reshape(rows,3)

    for j in range(1,3):
        for i in range(rows):
            c1 = np.round(time_gcc[i,j]/time_gcc[i,0],3)
            c2 = np.round(time_clang[i,j]/time_clang[i,0],3)
            c3 = np.round(time_icc[i,j]/time_icc[i,0],3)

            c4 = np.round(mem_gcc[i,j]/mem_gcc[i,0],3)
            c5 = np.round(mem_clang[i,j]/mem_clang[i,0],3)
            c6 = np.round(mem_icc[i,j]/mem_icc[i,0],3)

            c7 = np.round(size_gcc[i,j]/mem_gcc[i,0],3)
            c8 = np.round(size_clang[i,j]/mem_clang[i,0],3)
            c9 = np.round(size_icc[i,j]/mem_icc[i,0],3)

            c10 = np.round(runtime_gcc[i,0]/runtime_gcc[i,j],3)
            c11 = np.round(runtime_clang[i,0]/runtime_clang[i,j],3)
            c12 = np.round(runtime_icc[i,0]/runtime_icc[i,j],3)

            if j==2 and i>2:
                c1, c4, c7, c10 = "-","-","-","-"

            print 7-i, "Index", "&", c1, "&", c2, "&", c3, "&", c4, "&", c5, "&", c6, \
                "&", c7, "&", c8, "&", c9, "&", c10, "&", c11, "&", c12, r"\\\hline"
        print 



if __name__ == "__main__":

    quantity=3
    folder = "/home/roman/Dropbox/2016_SIMD_Paper/figures/Benchmarks_Compilation/"
    if quantity==0:
        figure_name = folder + "/Compilation_Time.png"
    if quantity==1:
        figure_name = folder + "/Memory_Usage.png"
    elif quantity==2:
        figure_name = folder + "/Binary_Size.png"
    elif quantity==3:
        figure_name = folder + "/Runtime.png"

    compilation_benchmark_plots(quantity=quantity)
    # compilation_benchmark_plots(quantity=quantity,save=True,figure_name=figure_name)


    # compilation_benchmark_tabulate()