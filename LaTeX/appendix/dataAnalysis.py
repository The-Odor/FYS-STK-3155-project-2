""" 
Welcome to the data analysis tool

I'm sorry you had to see this

please don't judge me too harshly, I am very very tired

the deadline is in 26 hours

My brain.... is mush

The tool functions. That is all I require.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




def sortingThingEta(instuff):
    if instuff[0] == "None":
        return 0
    else:
        return 1

def sortingThingEtaButNowForPartD(instuff):
    if instuff[0] == "None":
        return [0, -instuff[1]]
    else:
        return [1, -instuff[1]]

def sortingThingBoth(instuff):
    lamb = -instuff[0]
    if instuff[1][0] == "None":
        return [lamb, 0]
    else:
        return [lamb, 1]


def getNeuralNetwork():
    for suffix in ["classification", "regression"]:
        print()
        data = {
        "NNuse" :           suffix,
        "score" :           [],
        "layers" :          [],
        "nodes" :           [],
        "epochN" :          [],
        "minibatchSize" :   [],
        "eta" :             [],
        "lmbd" :            [],
        "alpha" :           [],
        "activFunct" :      [],
        "outFunc" :         []
        }

        with open("NeuralNetworkParameters_" + suffix + ".txt", "r") as infile:
            for _ in range(11):
                infile.readline()

            while True:
                line = infile.readline().replace(" ", "")
                if line.startswith("Optimal"):
                    break

                line = line.split("|")
                data["score"].append(float(line[0]))
                params = line[1].split(",")
                i = 0
                while i < len(params):
                    param = params[i]
                    name, val = param.split(":")
                    if param.startswith("eta"):
                        val = val[1:]
                        if val != "None":
                            val = float(val)
                        i+=1
                        data["eta"].append( [val, float(params[i][:-1])] )
                    else:
                        try:
                            data[name].append(float(val))
                        except ValueError:
                            data[name].append(val)
                    i+=1

            for i in data["layers"]:
                pass
                # print(i)
            # print(line)

        yield data



def getSGD():
    with open("SGDparameters.txt", "r") as infile:
        for _ in range(3):
            infile.readline()

        data = {
            "score" :           [],
            "MSE" :             [],
            "dataN" :           [],
            "polydegree" :      [],
            "minibatchSize" :   [],
            "epochN" :          [],
            "eta" :             [],
            "drag" :            []
        }

        for line in infile:
            # line = infile.readline()[1:-2].replace(" ", "")
            line = line[1:-2].replace(" ", "")
            values = line.split(",")

            data["score"].append(float(values[0]))
            data["MSE"].append(float(values[1]))
            data["dataN"].append(int(values[2]))
            data["polydegree"].append(int(values[3]))
            data["minibatchSize"].append(int(values[4]))
            data["epochN"].append(int(values[5]))
            data["eta"].append([(values[6][1:]), (values[7][:-1])])
            data["drag"].append(float(values[8]))


        yield data




    with open("SGDparameters_analytical.txt", "r") as infile:
        infile.readline()

        data = {
            "score" : [],
            "lmbd"  : []
        }


        # Reads Ridge
        while True:
            line = infile.readline()[:-1]
            if not line:
                break
            line = line.replace(" ", "").split("|")
            data["score"].append(float(line[1]))
            data["lmbd"].append(float(line[0]))


        # Reads last line of OLS
        line = infile.readline()[:-1].replace(" ", "").split("|")
        data["score"].append(float(line[1]))
        data["lmbd"].append((line[0]))


        yield data




def getLogistic():
    with open("LogisticRegressionParameters.txt", "r") as infile:
        for _ in range(3):
            infile.readline()


        data = {
            "score" :           [],
            "eta" :             [],
            "epochN" :          [],
            "minibatchSize" :   [],
            "drag" :            [],
            "l2" :              []
        }

        while True:
        # for _ in range(5):
            line = infile.readline()[:-1]
            if not line:
                break
            line = line.replace("[","").replace("]","").replace(" ","").split(",")
            data["score"].append(float(line[0]))
            data["eta"].append([float(line[1]), float(line[2])])
            data["epochN"].append(int(line[3]))
            data["minibatchSize"].append(int(line[4]))
            data["drag"].append(float(line[5]))
            data["l2"].append(float(line[6]))

        yield data        






def partA():
    inData = getSGD()
    dataNum = next(inData)
    dataAna = next(inData)


    for data, name in [(dataNum, "eta"), (dataAna, "lmbd")]:
        print()
        scores = data["score"]
        params = data[name]
        averages = {}
        medians  = {}
        numberOfPointsForEachThing = {}
        dataN = len(scores)
        for i in range(dataN):
            if np.isnan(scores[i]):
                if name=="eta":
                    try:
                        numberOfPointsForEachThing[tuple(params[i])] += 1
                    except KeyError:
                        numberOfPointsForEachThing[tuple(params[i])] = 1

            else:
                try:
                    averages[tuple(params[i])] += scores[i]
                    medians[tuple(params[i])].append(scores[i])
                    numberOfPointsForEachThing[tuple(params[i])] += 1
                except KeyError:
                    averages[tuple(params[i])] = scores[i]
                    medians[tuple(params[i])] = [(scores[i])]
                    numberOfPointsForEachThing[tuple(params[i])] = 1
                except TypeError:
                    try:
                        averages[params[i]] += scores[i]
                        medians[params[i]].append(scores[i])
                        numberOfPointsForEachThing[params[i]] += 1
                    except KeyError:
                        averages[params[i]] = scores[i]
                        medians[params[i]] = [(scores[i])]
                        numberOfPointsForEachThing[params[i]] = 1

        for key in averages:
            averages[key] /= len(medians[key]) 

        if name=="eta":
            print("Eta-values | Average | Median | Maximum | Fraction of functionally trained NN's")
            for key in averages:
                print(key, averages[key], medians[key][len(medians[key])//2], max(medians[key]), len(medians[key])/numberOfPointsForEachThing[key], sep=" | ")
        elif name=="lmbd":
            print("Lambda-value | score")
            for key in averages:
                print(key, averages[key], sep=" | ")

            lamX = []
            lamY = []
            del averages[('O', 'L', 'S')]
            averages = {key: averages[key] for key in sorted(averages.keys(), key=lambda x:-x)}
            for key in averages:
                if key != ('O', 'L', 'S'):
                    lamX.append(np.log(key))
                    lamY.append(averages[key])
            
            plt.plot(lamX, lamY)
            plt.xlabel(r"$\log_{10}(\lambda)$")
            plt.ylabel("R2-score")
            plt.title(r"R2-score of Ridge plotted logarithmically as a function of $\lambda$")
            plt.savefig("../images/partA_ridge.png")
            plt.show()


        










def partB():
    inData = getNeuralNetwork()
    dataClass = next(inData)
    dataRegre = next(inData)
    for data in [dataRegre]:
        scores = data["score"]
        lmbd  = data["lmbd"]
        eta   = data["eta"]

        dataN = len(scores)

        eta_dict = {}
        lmbd_dict = {}
        both_dict = {}

        for i in range(dataN):
            score = scores[i]
            if np.isnan(score):
                continue
            try:
                both_dict[(lmbd[i], tuple(eta[i]))].append(score)
                eta_dict[tuple(eta[i])].append(score)
                lmbd_dict[lmbd[i]].append(score)
            except KeyError:
                try:
                    eta_dict[tuple(eta[i])].append(score)
                    lmbd_dict[lmbd[i]].append(score)
                except KeyError:
                    eta_dict[tuple(eta[i])] = [score]
                    lmbd_dict[lmbd[i]] = [score]
                both_dict[(lmbd[i], tuple(eta[i]))] = [score]


        eta_dict  = {key: eta_dict[key]  for key in sorted(eta_dict.keys(),  key=sortingThingEtaButNowForPartD)}
        both_dict = {key: both_dict[key] for key in sorted(both_dict.keys(), key=sortingThingBoth)}
        lmbd_dict = {key: lmbd_dict[key] for key in sorted(lmbd_dict.keys(), key=lambda x:-x)}

        print("\n\n", data["NNuse"])
        for thing, nem in [(eta_dict, "eta"), (lmbd_dict, "lmbd"), (both_dict, "lambda, eta")]:
            print("\n{} | Average | Median | Maximum | Fraction of functionally trained SGD's".format(nem))
            for key in thing:
                print(key, np.average(thing[key]), sorted(thing[key])[len(thing[key])//2], max(thing[key]), len(thing[key])/dataN*len(thing), sep=" | ")

        etaN = len(eta_dict)
        lmbdN = len(lmbd_dict)


        plottingThing = np.zeros((lmbdN, etaN))
        for l, lkey in zip(range(lmbdN), lmbd_dict.keys()):
            for e, ekey in zip(range(etaN), eta_dict.keys()):
                plottingThing[l, e] = np.max(both_dict[tuple((lkey, ekey))])

        sns.set()
        _, ax = plt.subplots(figsize = (etaN, lmbdN))      
        sns.heatmap(plottingThing.T, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Maximum R2 score reached for Neural Network (regression) as function of eta and lambda")
        ax.set_ylabel(r"Varying value; $log_{10}(t_1)+7$ | Constant value; $-log_{10}(\eta)$ ")
        ax.set_xlabel(r"$-(1+log_{10}(\lambda))$")  

        plt.savefig("../images/partB.png")
        plt.show()
    # TODO: NN regression; regularization parameters (I assume that may be lambda and eta??), make an analysis







def partC():
    inData = getNeuralNetwork()
    dataClass = next(inData)
    dataRegre = next(inData)
    
    for data in [dataRegre]:
        methods = set(data["activFunct"])
        methodComparison = {meth: [] for meth in methods}
        
        scores = data["score"]
        dataN = len(scores)

        for i in range(dataN):
            if np.isnan(scores[i]):
                pass
            else:
                methodComparison[data["activFunct"][i]].append(scores[i])

        thing = methodComparison
        print("\n{} | Average | Median | Maximum | Fraction of functionally trained SGD's".format("Function"))
        for key in thing:
            print(key, np.average(thing[key]), sorted(thing[key])[len(thing[key])//2], max(thing[key]), len(thing[key])/dataN*len(methodComparison), sep=" | ")



    # TODO: NN regression; Compare Sigmoid, ReLU, and leaky ReLU.








def partD():
    inData = getNeuralNetwork()
    dataClass = next(inData)
    # dataRegre = next(inData)

    for data in [dataClass]:
        methods = list(set(data["activFunct"]))
        etas = list(sorted( set([tuple(i) for i in data["eta"]]), key = sortingThingEtaButNowForPartD))
        lmbd = list(set(data["lmbd"]))

        scores = data["score"]
        dataN = len(scores)

        plot_dict = {(method, tuple(eta), lmb): [] for method in methods for eta in etas for lmb in lmbd}

        for i in range(dataN):
            score = scores[i]
            if np.isnan(score):
                pass
            else:
                method = data["activFunct"][i]
                eta = data["eta"][i]
                lmb = data["lmbd"][i]
                plot_dict[(method, tuple(eta), lmb)].append(score)


        # etasMeshReference will be the index for etas bc fuckin apparently if you meshgrid a tuple
        # with a string "None" and float 0.1, then they are both fucking strings inside a list.
        # Fuck me, amirite
        etasMeshReference, lmbdMesh = np.meshgrid(range(len(etas)), lmbd)

        plottingThing = np.zeros((len(methods), etasMeshReference.shape[0], etasMeshReference.shape[1]))

        for m, meth in enumerate(methods):
            for i in range(etasMeshReference.shape[0]):
                for j in range(etasMeshReference.shape[1]):
                    etasIndexThing = tuple(etas[etasMeshReference[i,j]])
                    plottingThing[m, i, j] = np.max(plot_dict[(method, etasIndexThing, lmbdMesh[i,j])])


        for i, meth in enumerate(methods):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            surf = ax.plot_surface(etasMeshReference, np.log(lmbdMesh), plottingThing[i], cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, label=str(i))
            
            ax.set_zlim(-0.10, 1.0)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            plt.title("Neural network classification accuracy-score using activation function {}\nas a function of eta and lambda".format(meth))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.xlabel("$\eta$ reference index")
            plt.ylabel("$\log_{10}(\lambda)$")
            plt.savefig("../images/partD_" + meth + ".png")
            plt.show()


        score  = data["score"]
        layers = [int(i) for i in data["layers"]]
        nodes  = [int(i) for i in data["nodes"]]

        layers_unique = set(layers)
        nodes_unique  = set(nodes)

        plottingThingButNowForLayersAndNodes = {(i, j): [] for i in layers_unique for j in nodes_unique}

        for i in range(len(score)):
            if np.isnan(score[i]):
                pass
            else:
                plottingThingButNowForLayersAndNodes[(layers[i], nodes[i])].append(score[i])

        layersMesh, nodesMesh = np.meshgrid(tuple(layers_unique), tuple(nodes_unique))

        shorterName = np.zeros_like(layersMesh, dtype=float)
        shorterName += 1

        for i in range(layersMesh.shape[0]):
            for j in range(layersMesh.shape[1]):
                a = np.max(plottingThingButNowForLayersAndNodes[(layersMesh[i,j], nodesMesh[i,j])])
                shorterName[i][j] = a

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(layersMesh, nodesMesh, shorterName, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, label=str(i))
        
        ax.set_zlim(-0.10, 1.0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        plt.title("Neural network classification accuracy-score as a function of layers and nodes")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel("Layers")
        plt.ylabel("Nodes")
        plt.savefig("../images/partD_layersandnodes.png")
        plt.show()


    # TODO: NN classification; Discuss learning rate, lambda, activation functions,
    #       number of hidden layers/nodes, also activation functions again?

    # That is:
    #   - learning rate
    #   - lambda
    #   - activation functions
    #   - number of hidden layers/nodes

    # Graph A: X 3D plots. x-dim=learningrate. y-dim=lambda. 1 plot per activation function. See how it feels





def partE():
    inData = getLogistic()
    dataClass = next(inData)

    for data in [dataClass]:
        etas = list(sorted( set([tuple(i) for i in data["eta"]]), key = sortingThingEtaButNowForPartD))
        l2s  = list(set(data["l2"]))

        scores = data["score"]
        dataN = len(scores)

        plot_dict = {(tuple(eta), l2): [] for eta in etas for l2 in l2s}

        for i in range(dataN):
            score = scores[i]
            if np.isnan(score):
                pass
            else:
                eta = data["eta"][i]
                l2  = data["l2"][i]
                plot_dict[(tuple(eta), l2)].append(score)


        # etasMeshReference will be the index for etas bc fuckin apparently if you meshgrid a tuple
        # with a string "None" and float 0.1, then they are both fucking strings inside a list.
        # Fuck me, amirite
        l2s = sorted(l2s)
        etasMeshReference, l2Mesh = np.meshgrid(range(len(etas)), l2s)

        plottingThing = np.zeros((etasMeshReference.shape[0], etasMeshReference.shape[1]))

        for i in range(etasMeshReference.shape[0]):
            for j in range(etasMeshReference.shape[1]):
                etasIndexThing = tuple(etas[etasMeshReference[i,j]])
                plottingThing[i, j] = np.max(plot_dict[(etasIndexThing, l2Mesh[i,j])])


        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(etasMeshReference, np.log(l2Mesh), plottingThing, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, label=str(i))
        
        ax.set_zlim(-0.10, 1.0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        plt.title("Logistic regression accuracy as a function of eta and l2")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel("$\eta$ reference index")
        plt.ylabel("$\log_{10}(l_2)$")
        plt.savefig("../images/partE_l2andeta.png")
        plt.show()

        print(etas, l2s)


    # TODO: LogReg; "add l2 regularization parameter lambda". 
    #       Compare with NN and sklearn

def partF():
    pass
    # TODO: I don't know whether this is should be here, but they ask for a 
    #       summary of the whole schebang, what algorithms work best for 
    #       regression and classification etc.

if __name__ == "__main__":
    
    # a = getNeuralNetwork()
    # data = next(a)
    # data = next(a)
    # print(data["score"])

    # b = getSGD()
    # data = next(b)
    # data = next(b)
    # print(data["score"])

    # c = getLogistic()
    # data = next(c)
    # for key in data:
    #     print(key, data[key][:10])

    partA()

    partB()

    partC()

    partD()

    partE()
    pass