#Tom Wallace
#6482558
#Brock University

import json
import os
from os.path import basename

import stumpy
import matplotlib
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib import pyplot as plt
import ruptures as rpt
import random

from bkpt import bkpt

filename = "timeseries\\"+random.choice(os.listdir("timeseries")) #Picks a random json benchmark file from \timeseries directory
print(filename)
#filename = "eclipse__eclipse-collections#org.eclipse.collections.impl.jmh.map.ChainMapPutTest.ec#isPresized=false&loadFactor=0.80f&size=3000000.json"
#filename = "timeseries\\h2oai__h2o-3#water.util.IcedHashMapBench.writeMap#arrayType=Array&array_values_length=10000&keyType=String&n_entries=100000&valueType=Boolean.json"
#filename = "timeseries\\JCTools__JCTools#org.jctools.jmh.latency.QueueBurstCost.burstCost#burstSize=100&consumerCount=1&qCapacity=132000&qType=SpmcArrayQueue&warmup=true.json"
def get_measurements(file): #Retrieves the 10 timeseries representing each fork of the benchmark from a given json filename
    with open(file) as f:
        data = f.read()
        meas = json.loads(data)
        # print(json.dumps(measurements, indent=4))
    return meas
def rptPelt(measurements): #Performs changepoint analysis and displays graph with changepoints
    points = np.array(measurements)
    # detection
    algo = rpt.Pelt(model="rbf", min_size=30).fit(points)
    result = algo.predict(20)
    rpt.display(points, result)
    plt.show()
    return result
def rptClassifyWithoutDisplay(measurements, m): #Performs changepoint analysis and prints classification to console
    points = np.array(measurements)
    # detection
    algo = rpt.Pelt(model="rbf", min_size=m/2).fit(points)
    result = algo.predict(20)
    print(classify(result,measurements))
def classify(points, measurements): #Classifies whether or not a benchmark is able to reach steady state or not by using changepoint analysis
    assert len(points)>0
    segs = []
    if len(points)==1:
        return "inconclusive"
    if len(points)==2:
        segs.append(bkpt(measurements[0:points[0]], points[0]))
        segs.append(bkpt(measurements[points[0]:points[1]], points[1]))
    else:
        i = 0
        pointsclone = list(points)
        pointsclone.append(0)
        pointsclone.sort()
        while i < len(pointsclone) - 1:
            segs.append(bkpt(measurements[pointsclone[i]:pointsclone[i+1]], pointsclone[i+1]))
            i=i+1
    last = segs[len(segs)-1]
    lower = last.mean-max(last.var, 0.001)
    higher = last.mean+max(last.var, 0.001)
    cls = "flat"
    i = len(segs)-2
    while i > -1:
        cur_seg = segs[i]
        i -= 1
        if cur_seg.mean + cur_seg.var >= lower and cur_seg.mean - cur_seg.var <= higher:
            continue
        elif cur_seg.end > len(segs) - 500:
            cls = "no steady state"
            break
        elif cur_seg.mean < lower:
            cls = "slowdown"
            break
        assert (cur_seg.mean > higher)
        cls = "warmup"
    return cls

def stumpyFindPattern(measurements, m): #Runs stumpy across all forks in a benchmark, identifies the best pattern match within one,
    #and compares this match with all other forks to see if the pattern can be found again.
    bestMatch = 999999
    bestMatchIndex = 0
    bestMatchRun = 0
    i=0
    for arr in measurements:
        mp = stumpy.stump(arr, m)
        motif_idx = np.argsort(mp[:, 0])[0]
        nearest_neighbor_idx = mp[motif_idx, 1]
        if mp[motif_idx,0] < bestMatch:
            bestMatch = mp[motif_idx, 0]
            bestMatchIndex = motif_idx
            bestMatchRun = i
        i=i+1
    plt.plot(measurements[bestMatchRun])
    plt.axvline(bestMatchIndex,
                c='red')
    plt.show()

    for arr in measurements:
        mp2 = stumpy.stump(T_A = measurements[bestMatchRun],
                        m = m,
                        T_B = arr,
                        ignore_trivial = False)
        motif_index2 = mp2[:, 0].argmin()
        motif_matchindex = mp2[motif_index2,1]
        plt.xlabel('Subsequence')
        plt.ylabel('Matrix Profile')
        plt.scatter(motif_index2,
                    mp2[motif_index2, 0],
                    c='red',
                    s=100)
        plt.plot(mp2[:, 0])
        plt.show()
        fig, axs = plt.subplots(2)
        axs[0].plot(measurements[bestMatchRun])
        axs[0].plot(arr)
        axs[1].plot(measurements[bestMatchRun][motif_index2: motif_index2 + m], label='Best Pattern')
        axs[1].plot(arr[motif_matchindex:motif_matchindex + m], label='Match to Compare')

        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Time')

        plt.legend()

        plt.show()


def stumpyRun(measurements, m): #Runs only Stumpy (Motif detection).
    mp = stumpy.stump(measurements, m)
    motif_idx = np.argsort(mp[:, 0])[0]
    nearest_neighbor_idx = mp[motif_idx, 1]

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

    axs[0].plot(measurements)
    rect = Rectangle((motif_idx, min(measurements)), m, sum(measurements)/3000, facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((nearest_neighbor_idx, min(measurements)), m, sum(measurements)/3000, facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Iterations', fontsize='18')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plt.show()

def stumpyRunWithRpt(measurements, m, i): #Runs Stumpy (Motif detection) and Ruptures (Changepoint) on the same graph.
    points = np.array(measurements)
    # detection
    algo = rpt.Pelt(model="rbf", min_size=m/2).fit(points)
    result = algo.predict(15)
    classification = classify(result, measurements)
    print(classification)
    mp = stumpy.stump(measurements, m)
    motif_idx = np.argsort(mp[:, 0])[0]
    nearest_neighbor_idx = mp[motif_idx, 1]
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
    fig.suptitle(filename, fontsize='12')
    axs[0].plot(measurements)
    rect = Rectangle((motif_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((nearest_neighbor_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    for f in result:
        axs[0].axvline(x=f, linestyle="dashed", color='r')
    axs[1].set_xlabel('Iterations', fontsize='18')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plotfile = basename(filename).replace('.json', '.png')
    plotfile = str(i) + plotfile
    if classification == "no steady state":
        plotfile = "RQ3\\SteadyState\\"+plotfile
    else:
        plotfile = "RQ3\\FlatOther\\" + plotfile

    plt.savefig(plotfile, dpi=300)
    plt.clf()

def stumpyJudgeMotif(measurements, m, i): #Runs Stumpy (Motif detection) with classification
    #Outputs a text file with min, max, variance and mean of Matrix Profile, and an image file of both graphs to folder \RQ1
    points = np.array(measurements)
    # detection
    algo = rpt.Pelt(model="rbf", min_size=m/2).fit(points)
    result = algo.predict(15)
    classification = classify(result, measurements)
    print(classification)
    mp = stumpy.stump(measurements, m)
    motif_idx = np.argsort(mp[:, 0])[0]
    nearest_neighbor_idx = mp[motif_idx, 1]
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
    fig.suptitle(filename, fontsize='12')
    axs[0].plot(measurements)
    rect = Rectangle((motif_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((nearest_neighbor_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Iterations', fontsize='18')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plotfile = basename(filename).replace('.json', '.png')
    plotfile = str(i) + plotfile
    if classification == "no steady state":
        plotfile = "RQ1\\NoSteadyState\\"+plotfile
    else:
        plotfile = "RQ1\\FlatOther\\" + plotfile
    plt.savefig(plotfile, dpi=300)
    plt.clf()
    textfile = plotfile.replace('.png','.txt')
    mini = min(mp[:, 0])
    maxi = max(mp[:, 0])
    var = np.var(mp[:, 0])
    mean = np.mean(mp[:, 0])
    f= open(textfile, 'w')
    f.writelines([str(mini)+"\n", str(maxi)+"\n", str(var)+"\n", str(mean)])
    f.close()

def findAnomaly(measurements, m, i): #Runs Stumpy (Motif detection), displaying discords and motifs on the graph
    #and prints out mean+stdev of the timeseries, its motifs, and the discord, as well as if any have a potential anomaly
    mp = stumpy.stump(measurements, m)
    motif_idx = np.argsort(mp[:, 0])[0]
    nearest_neighbor_idx = mp[motif_idx, 1]
    discord_idx = np.argsort(mp[:, 0])[-1]

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

    mean = np.mean(measurements)
    stdev = np.std(measurements)
    m1_mean = np.mean(measurements[motif_idx:motif_idx+m])
    m1_std = np.std(measurements[motif_idx:motif_idx+m])
    print (str(i)+": "+str(mean)+", "+str(stdev))
    print(str(i) + "  motif at "+str(motif_idx)+":" + str(m1_mean) + ", " + str(m1_std))
    m2_mean = np.mean(measurements[nearest_neighbor_idx:nearest_neighbor_idx + m])
    m2_std = np.std(measurements[nearest_neighbor_idx:nearest_neighbor_idx + m])
    print(str(i) + "  motif at " + str(nearest_neighbor_idx) + ":" + str(m2_mean) + ", " + str(m2_std))
    dis_mean = np.mean(measurements[discord_idx:discord_idx + m])
    dis_std = np.std(measurements[discord_idx:discord_idx + m])
    print(str(i) + "  discord at " + str(discord_idx) + ":" + str(dis_mean) + ", " + str(dis_std))
    if motif_idx >30:
        if m1_mean>mean+(2*stdev):
            print("Most likely anomaly at first motif")
        elif m1_mean>mean+(stdev):
            print("Potential anomaly at first motif")
        if m1_std>stdev*2:
            print("Spike at first motif")
    if nearest_neighbor_idx > 30:
        if m2_mean>mean+(2*stdev):
            print("Most likely anomaly at second motif")
        elif m2_mean>mean+(stdev):
            print("Potential anomaly at second motif")
        if m2_std>stdev*2:
            print("Spike at second motif")
    if discord_idx > 30:
        if dis_mean>mean+(2*stdev):
            print("Most likely anomaly at discord")
        elif dis_mean>mean+(stdev):
            print("Potential anomaly at discord")
        if dis_std>stdev*2:
            print("Spike at discord")

    axs[0].plot(measurements)
    rect = Rectangle((motif_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((nearest_neighbor_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((discord_idx, min(measurements)), m, max(measurements), facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Iterations', fontsize='18')
    axs[1].set_ylabel('Matrix Profile', fontsize='20')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
    axs[1].axvline(x=discord_idx, linestyle="dashed", color="red")
    axs[1].plot(mp[:, 0])
    plt.show()


measurements = get_measurements(filename)

i=0
for arr in measurements:
    #stumpyRunWithRpt(arr, 300, i) #
    #stumpyJudgeMotif(arr, 300, i)
    findAnomaly(arr,200,i)
    i=i+1
    #a = rptPelt(arr)
    #print(classify(a, arr))



