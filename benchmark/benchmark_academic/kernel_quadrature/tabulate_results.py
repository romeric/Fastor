import numpy as np



def read():

    filename = "/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/all_results2"
    with open(filename) as f:
        lines = f.readlines()

    # print lines

    each_bench = []
    for counter, line in enumerate(lines):
        # print line
        if "cd" in line:
            each_bench.append(counter)
            # print line, counter

    # print each_bench
    models = ["2MR", "2MRDS", "2El1", "2El1DS", "2El2", "2El2DS",
                "3MR", "3MRDS", "3El1", "3El1DS", "3El2", "3El2DS"]
    all_speed_ups = {}
    for i in range(len(each_bench)-0):
        # print models[i]
        counter = 0
        timer = []
        # print each_bench[i],each_bench[i+1]
        # for j in range(each_bench[i]+2,each_bench[i+1]-1):
        for j in range(each_bench[i]+2,each_bench[i]+27):
            if counter % 2 != 0:
                lists = lines[j].split(" ")
                if lists[7]!='ms.' and lists[7]!='s.':
                    lists[6] = 10.**(-6)*float(lists[6]) 
                elif lists[7]=='ms.':
                    lists[6] = 10.**(-3)*float(lists[6]) 
                if lists[7]=='s.':
                    lists[6] = float(lists[6])  
                # print unicode(lists[7])
                # print(lists[7].encode('UTF-8'))
                # print lists[7]
                # timer.append(float(lists[6]))
                timer.append(lists[6])
                # print lists[7]  
            counter+=1
        timer = np.array(timer)
        speedup = timer[::2]/timer[1::2]
        all_speed_ups[models[i]] = speedup
        # exit()

    # print all_speed_ups

    for p in range(1,7):
        print "$p=$"+str(p), "&", np.around(all_speed_ups["2MR"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["2El1"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["2El2"][p-1], decimals=3), "\\\ \hline"
    print 
    for p in range(1,7):
        print "$p=$"+str(p), "&", np.around(all_speed_ups["3MR"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["3El1"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["3El2"][p-1], decimals=3), "\\\ \hline"
    print 
    for p in range(1,7):
        print "$p=$"+str(p), "&", np.around(all_speed_ups["3MRDS"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["3El1DS"][p-1], decimals=3), "&", \
                np.around(all_speed_ups["3El2DS"][p-1], decimals=3), "\\\ \hline"


read()