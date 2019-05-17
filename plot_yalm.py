import yaml
import matplotlib.pylab as plt
import numpy as np

boosters = ['AdaBooster', 'OGBooster', 'EXPBooster', 'OCPBooster', 'OnlineBBM', 'OSBooster', 'AdaBoosterOL']
numbers=[10,50,100,200,300,500,1000]

for num in numbers:
    fig = plt.figure()
    ax = plt.subplot(111)
    results = []
    for id,booster in enumerate(boosters):
        with open(booster + '_Perceptron_au_0_0_0.yml', 'r') as stream:
            data_loaded = yaml.load(stream)

        results.append(data_loaded['accuracy'][num][-1])
        xnew = np.linspace(0, 690, 690)
        if id<2:
            ax.plot(data_loaded['accuracy'][num],  '--', label=booster, linewidth=2)
        else:
            ax.plot(data_loaded['accuracy'][num], label=booster, linewidth=2)
    ax.legend(loc='lower left', bbox_to_anchor= (0.0, 0.9),
          ncol=3, fancybox=True, shadow=True)
    #plt.title('Number of weak learners: %.02f' %num)
    #print results
    plt.savefig('../' + str(num))


exit(0)
boosters = ['AdaBooster', 'EXPBooster', 'OCPBooster', 'OGBooster', 'OnlineBBM', 'AdaBoosterOL']

fig = plt.figure()
ax = plt.subplot(111)

for booster in boosters:
    with open(booster + '_Perceptron_10_100_10.yml', 'r') as stream:
        data_loaded1 = yaml.load(stream)

    with open(booster + '_Perceptron_100_1000_100.yml', 'r') as stream:
        data_loaded2 = yaml.load(stream)

    list1 = sorted(data_loaded1['accuracy'].items()) # sorted by key, return a list of tuples
    list2 = sorted(data_loaded2['accuracy'].items()) # sorted by key, return a list of tuples
    lists = list1 + list2
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax.plot(x, y, label=booster, marker='o', linewidth=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=3, fancybox=True, shadow=True)
plt.show()
