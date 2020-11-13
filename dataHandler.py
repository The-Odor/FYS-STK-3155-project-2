import numpy as np
import math as mt
from miscFunctions import to_categorical_numpy, defaultingFunc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class dataHandler():
    def __init__(self):
        self.N = None # dataN refers to actual datapoints, and is N^2
        self.dataN = None # Datapoints along a single axis
        self.minibatchSize = None
        self.minibatchN = None
        self.epochN = None

        self.p = None

        self.x = None
        self.y = None
        self.z = None

        self.zTrue = None
        self.X = None

        self.nfeatures = None
        self.noisefactor = None

        self.modellingFunction = None


    def FrankeFunction(self, x,y):
        """Function modelled in the by the SGD class
        The Franke Function is two-dimensional and is comprised of four 
        natural exponents

        Args:
            x (np.ndarray): array of floats between 0 and 1. x-dimension
            y (np.ndarray): array of floats between 0 and 1. y-dimension

        Returns:
            (numpy.ndarray) : resultant array from Frankefunction
        """
        term1 = 0.75*np.exp(-(9*x-2)**2/4.00 - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-(9*x+1)**2/49.0 - 0.10*(9*y+1))
        term3 = 0.50*np.exp(-(9*x-7)**2/4.00 - 0.25*((9*y-3)**2))
        term4 =-0.20*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4


    def craftDataset(self, N=None, minibatchSize=None, epochN=None, noisefactor=None, modellingFunction=None):
        """Crafts dataset. Can be used to overwrite values for noisefactor, dataN, p, minibatchSize, 
        and epochN if these values are given as arguments.

        Creates two linearly spaced datasets between 0 and 1 of size dataN/2, 
        then creates two uniformly random datasets between 0 and 1 of size 
        dataN/2. These two sets of datasets are concatenated together to form 
        the x and y datasets, which are put in a meshgrid and flattened to get
        a good spread of data.

        The response values are then generated using the modellingFunction, 
        which is FrankeFunction by default.

        Args:
            N (int, optional): square root of the number of datapoints used in total. Defaults to 100.
            minibatchSize (int, optional): amount of datapoints used in each iteration. Defaults to 64.
            epochN (int, optional): number of times dataset is iterated through. Defaults to 500.
            noisefactor (float, optional): noise as a fraction of mean data added to data. Defaults to 0.1.
            modellingFunction (function, optional): function used for modelling. Defaults to self.Frankefunction.
        """

        self.noisefactor   = defaultingFunc(self.noisefactor, noisefactor, 0.0)
        self.dataN         = defaultingFunc(self.dataN, N, 100)
        self.minibatchSize = defaultingFunc(self.minibatchSize, minibatchSize, 64)
        self.epochN        = defaultingFunc(self.epochN, epochN, 500)

        self.N = self.dataN**2
        self.minibatchN = int(self.N/self.minibatchSize)

        # Generates linearly and randomly spaced datasets
        x1 = y1 = np.linspace(0, 1, mt.ceil(self.dataN/2))
        x2 = np.random.uniform(0, 1, mt.floor(self.dataN/2))
        y2 = np.random.uniform(0, 1, mt.floor(self.dataN/2))

        # Contatenates datasets, makes a meshgrid, and flattens them
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        x, y = np.meshgrid(x,y)
        self.x = x.flatten()
        self.y = y.flatten()

        # Applies modelling function and generates expected values
        self.modellingFunction = defaultingFunc(self.modellingFunction, modellingFunction, self.FrankeFunction)
        self.zTrue = self.modellingFunction(self.x, self.y)
        self.z = self.zTrue + self.noisefactor*np.random.randn(self.N)*self.zTrue.mean()
        self.z = self.z.reshape(-1,1)


    def makePolynomial(self, p=None, scaling=None):
        """Crafts the design matrix X as a scaled two-dimensional polynomial of 
        x and y, as defined by craftDataset()

        Will call craftDataset if self.x, self.y, or self.z are not yet defined
        and use default values

        Args:
            p (int, optional): polynomial degree used. Defaults to 8.
            scaling (bool, optional): If true scales design matrix by subtracting average. Defaults to True.
        """
        self.p = defaultingFunc(self.p, p, 8)
        if scaling is None: scaling = True

        if self.x is None or self.y is None or self.z is None: 
            print("in dataHandler.makePolynomial(): using defaulting self.craftDataset() due to self.x, self.y, or self.z not being defined")
            self.craftDataset()

        self.nfeatures = int(((self.p+1)*(self.p+2))/2)
        self.X = np.zeros((len(self.x), self.nfeatures))

        ind = 0
        for i in range(self.p+1):
            for j in range(self.p+1-i):
                self.X[:,ind] = self.x**i * self.y**j
                ind += 1

        if scaling:
            self.X[:,1:] -= np.mean(self.X[:,1:], axis=0)


    def makeImageLabel(self):
        """Imports dataset from sklearn.datasets. Uses images and labels from
        datasets.load_digits()
        """
        from sklearn import datasets
        digits = datasets.load_digits()
        digits.images = digits.images; digits.target = digits.target
        images = digits.images
        labels = digits.target

        self.baseX = images
        self.basez = labels

        images = images.reshape(len(images), -1)
        labels = labels.reshape(len(labels), -1)
        labels = to_categorical_numpy(labels)

        self.X = images
        self.z = labels

    def plotRealAgainstPredicted(self, trainedNetwork):
        """
        Plots predicted values against true values self.z to compare
        them and visually affirm the validity of the model

        Args:
            trainedNetwork (neural network): Trained neural network to predict datapoints
        """

        if self.z is None:
            print("in dataHandler.plotRealAgainstPredicted: using defaulting self.crafDataset() due to self.z not being defined")
            self.craftDataset()

        trainedNetwork.giveInput(self.X, self.z)
        trainedNetwork.predict()

        # Plotting basis for function
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        x, y = np.meshgrid(x,y)

        z = self.modellingFunction(x, y)
        z_ = trainedNetwork.prediction

        fig = plt.figure()
        ax = fig.gca(projection="3d")
 
        cm.coolwarm = cm.coolwarm
        # Visual Studio Code complains that cm.coolwarm doesn't exist
        # unless I put in this line, so here we are

        # Limit the amount of plotted points to not overwork the plotter
        Npoints = 1000
        plotIndex = np.random.choice(range(len(self.x)), Npoints, replace=False)
        x_plot = self.x[plotIndex]
        y_plot = self.y[plotIndex]
        z_plot = z_[plotIndex]
       
        # Plot the surface and scatterplot.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.scatter(x_plot, y_plot, z_plot)

        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        plt.title("Prediction on {}".format(self.modellingFunction.__name__))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def printPredictions(self, trainedNetwork):
        """Plots images with expected and predicted labels

        Args:
            trainedNetwork (neural network): Trained neural network to predict labels
        """

        indices = np.arange(trainedNetwork.Ninputs)
        Nimages = 5
        plotIndices = np.random.choice(indices, size=Nimages)

        X = self.X[plotIndices]
        z = self.z[plotIndices]
        images = self.baseX[plotIndices]
        labels = self.basez[plotIndices]

        trainedNetwork.giveInput(X,z)
        trainedNetwork.predict()
        predicted = trainedNetwork.predictedLabel


        for i, image in enumerate(images):
            # plt.subplot(1, Nimages*2, 2*i+1)
            plt.subplot(1, Nimages, i+1)
            plt.axis('off')
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            if labels[i] == predicted[i]:
                plt.title("Correct!\nLabel: %d\nPredicted %d" % (labels[i], predicted[i]))
            else:
                plt.title("Wrong.\nLabel: %d\nPredicted %d" % (labels[i], predicted[i]))
        plt.show()