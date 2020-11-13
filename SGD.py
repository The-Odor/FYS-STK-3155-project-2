import numpy as np
import math  as mt
from matplotlib.pyplot import plot, axis, xlabel, ylabel, title, show, legend, scatter
from miscFunctions import defaultingFunc, MSE, R2
# from sklearn.linear_model import SGDRegressor


class SGDR():
    """
    Stochastic Gradient Descent-Regression class
    """

    def __init__(self):

        self.N               = None # Datapoints along a single axis
        self.dataN           = None # dataN refers to actual datapoints, and is N^2
        self.minibatchSize   = None
        self.minibatchN      = None
        self.epochN          = None

        self.p = None

        self.x = None
        self.y = None
        self.z = None

        self.zTrue = None
        self.theta = None
        self.X     = None

        self.nfeatures   = None
        self.noisefactor = None

        self.stepLength         = None
        self.drag               = None
        self.lamb               = None
        self.modellingFunction  = None

        self.MeanSquareError    = None
        self.R2Score            = None

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
        

    def defineStepLength(self, t0=None, t1=None):
        """
        stepLength = lambda t: t0/t+t1
        stepLength may also be constant if either only t0 or only t1 is given

        Args:
            t0 (float, optional): nominator. Defaults to 10.
            t1 (float, optional): added with t to denominator. Defaults to 50.
        """
        if t0 is None and t1 is None:
            t0 = 10
            t1 = 50
            self.stepLength = lambda t: t0/(t+t1)
        elif t1 is None:
            self.stepLength = lambda t: t0
        elif t0 is None:
            self.stepLength = lambda t: t1
        else:
            self.stepLength = lambda t: t0/(t+t1)



    def craftDataset(self, N=None, p=None, minibatchSize=None, epochN=None, noisefactor=None, modellingFunction=None):
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
            p (int, optional): polynomial degree used. Defaults to 5.
            minibatchSize (int, optional): amount of datapoints used in each iteration. Defaults to 64.
            epochN (int, optional): number of times dataset is iterated through. Defaults to 500.
            noisefactor (float, optional): noise as a fraction of mean data added to data. Defaults to 0.1.
            modellingFunction (function, optional): function used for modelling. Defaults to self.Frankefunction.
        """
        self.dataN         = defaultingFunc(self.dataN, N, 100)
        self.p             = defaultingFunc(self.p, p, 8)
        self.minibatchSize = defaultingFunc(self.minibatchSize, minibatchSize, 64)
        self.epochN        = defaultingFunc(self.epochN, epochN, 500)
        self.noisefactor   = defaultingFunc(self.noisefactor, noisefactor, 0.1)

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


    def craftX(self, scaling=None):
        """Crafts the design matrix X as a scaled two-dimensional polynomial of 
        x and y, as defined by craftDataset()

        Will call craftDataset if self.x, self.y, or self.z are not yet defined
        and use default values

        Args:
            scaling (bool, optional): If true scales design matrix by subtracting average. Defaults to True.
        """
        if scaling is None: scaling = True

        if self.x is None or self.y is None or self.z is None: 
            print("in SGDR.craftX(): using defaulting self.craftDataset() due to self.x, self.y, or self.z not being defined")
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


    def SGD(self):
        """
        The main Stochastic Gradient Descent method

        Generates a random theta using numpy.random.rand(self.nfeatures,1) and 
        adjusts it using the SGD method

        Will call craftX() if self.X is not yet defined and defineStepLength() 
        if self.stepLength is not yet defined, using default values for each
        """
        if self.X is None: 
            print("in SGDR.SGD(): using defaulting self.craftX() due to self.X not being defined")
            self.craftX()

        if self.stepLength is None: 
            print("in SGDR.SGD(): using defaulting self.defineStepLength() due to self.stepLength not being defined")
            self.defineStepLength()
            
        # Theta initially chosen to be completely random
        self.theta = np.random.rand(self.nfeatures, 1) 

        indexSelectionPool = range(self.N)
        inertia = np.zeros_like(self.theta)
        drag = 1
        # drag is not really drag, but rather the inverse, where drag=1 means 
        # that no momentum is lost drag=0 means that there never is momentum
        # TODO: define drag so that is can be controlled outside the function
        #       for benchmarking purposes, etc.

        # plotlist = [] # DEBUG
        # plotepoch = [] # DEBUG

        for epoch in range(self.epochN):
            batch = np.random.choice(indexSelectionPool, self.minibatchSize, replace=False)
            for batchIndex in range(self.minibatchN):

                xi = self.X[batch].reshape(self.minibatchSize,-1)
                zi = self.z[batch].reshape(-1,1) 
                # .reshape gets correct shape if self.minibatchSize = 1, which
                #  will result in floats instead of arrays when using the @ operator

                gradient = xi.T @ ((xi @ self.theta) - zi)
                gradient = gradient.sum(axis=1).reshape(-1, 1) * (2 / self.minibatchSize)

                step = self.stepLength(epoch*self.minibatchN + batchIndex)

                inertia = drag*inertia
                gradient += inertia

                self.theta -= gradient*step

        #     plotlist.append(R2(self.X @ self.theta, self.z)) # DEBUG
        #     plotepoch.append(epoch) # DEBUG

        # import matplotlib.pyplot as plt # DEBUG
        # plt.plot(plotepoch,plotlist) # DEBUG
        # plt.xlabel("Number of epochs")
        # plt.ylabel("R2-score")
        # plt.title("SGD R2 over a number of epochs as optimal parameters")
        # plt.savefig("../images/partA_R2.png")
        # plt.show() # DEBUG

        self.MeanSquareError = MSE(self.X @ self.theta, self.z)
        self.R2Score = R2(self.X @ self.theta, self.z)


    def OLS(self):
        """Analytical OLS solution that can be used to compare results
        """
        self.theta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.z
        self.MeanSquareError = MSE(self.X @ self.theta, self.z)
        self.R2Score = R2(self.X @ self.theta, self.z)

    def ridge(self, lamb):
        """Analytical Ridge solution that can be used to compare results

        Args:
            lamb (float): Value of diagonal matrix added into OLS solution
        """
        self.theta = np.linalg.pinv(self.X.T @ self.X + lamb*np.identity(self.nfeatures)) @ self.X.T @ self.z
        self.MeanSquareError = MSE(self.X @ self.theta, self.z)
        self.R2Score = R2(self.X @ self.theta, self.z)


    def plotRealAgainstPredicted(self):
        """
        Plots predicted values self.theta against true values self.z to compare
        them and visually affirm the validity of the model
        """
        if self.z is None:
            print("in SGDR.plotRealAgainstPredicted: using defaulting self.crafDataset() due to self.z not being defined")
            self.craftDataset()

        if self.theta is None: 
            print("in SGDR.plotRealAgainstPredicted: using defaulting self.SGD() due to self.theta not being defined")
            self.SGD()


        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import figure
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        fig = figure()
        ax = fig.gca(projection="3d")

        # Make data.
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        x, y = np.meshgrid(x,y)

        z = self.modellingFunction(x, y)
        z_ = self.X @ self.theta

        cm.coolwarm = cm.coolwarm
        # Visual Studio Code complains that cm.coolwarm doesn't exist
        # unless I put in this line, so here we are

        # Plot the surface and scatterplot.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

        # Makes sure not too many points are plotted for matplotlib.pyplot
        Npoints = np.min((len(z_), 1000))
        plotIndex = np.random.choice(range(len(self.x)), Npoints, replace=False)
        x_plot = self.x[plotIndex]
        y_plot = self.y[plotIndex]
        z_plot = z_[plotIndex]
       
        ax.scatter(x_plot, y_plot, z_plot)

        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        show()










    
if __name__ == "__main__":
    regressor = SGDR()
    regressor.defineStepLength(10, 1)
    regressor.drag = 0
    regressor.craftDataset(200, 15, 100, 250)
    regressor.craftX()
    regressor.SGD()
    print(regressor.R2Score)
    regressor.plotRealAgainstPredicted()


    """
    Following is code for optimizing parameters for a set of parameters.
    """

    testSGD = False
    testANA = False
    writeToFile = False

    # Parameters

    # SGD
    N_ = [50,100]
    p_ = [5,8,10,12,15]
    minibatchSize_ = [10,50,100]
    epochN_ = [250,500]
    # t0_t1_ = [(10, 10), (10, 50), (10, 1), (None, 0.1), (None, 0.01), (None, 0.001)]
    t0_t1_ = [(None, 0.1), (None, 0.01), (None, 0.001)]
    dragSpace = np.linspace(0,1,5)

    # Ridge
    lambSpace = np.concatenate((np.logspace(0,-4, 100), np.array([0])))


    if testSGD:
        # SGD SOLUTION
        results = []
        progressCounter = 0
        progressMax = len(N_)*len(p_)*len(minibatchSize_)*len(epochN_)*len(t0_t1_)*len(dragSpace)
        for N in N_:
            for p in p_:
                for minibatchSize in minibatchSize_:
                    for epochN in epochN_:
                        for t0, t1 in t0_t1_:
                            for drag in dragSpace:
                                regressor = SGDR()
                                regressor.defineStepLength(t0, t1)
                                regressor.drag = drag
                                regressor.craftDataset(N, p, minibatchSize, epochN)
                                regressor.craftX()
                                regressor.SGD()

                                results.append([regressor.R2Score,
                                                regressor.MeanSquareError,
                                                N,
                                                p,
                                                minibatchSize,
                                                epochN,
                                                (t0, t1),
                                                drag])
                                
                                progressCounter += 1
                                print(progressCounter, "/", progressMax)


        results.sort(key=(lambda k: -k[0]))
        for i in results[:10]:
            print(i)

        if writeToFile:
            with open("parameterValues/SGDparameters.txt", "w") as outfile:
                outfile.write("values: R2 | MSE | # of datapoints | Polynomial degree | minibatch size | # of epochs | t0, t1 | Momentum-coefficient\n\n\n")
                for i in results:
                    outfile.write(str(i) + "\n")

        N, p, minibatchSize, epochN, t0t1, drag = results[0][2:]
        t0, t1 = t0t1
        regressor.defineStepLength(t0, t1)
        regressor.drag = drag
        regressor.craftDataset(N, p, minibatchSize, epochN)
        regressor.craftX()
        regressor.SGD()
        regressor.plotRealAgainstPredicted()





    if testANA:
        N = 200
        p = 15


        # RIDGE SOLUTION
        lambScore = np.zeros((len(lambSpace), 2))

        for i, lamb in enumerate(lambSpace):
            regressor = SGDR()
            regressor.craftDataset(N, p, None, None)
            regressor.craftX()
            regressor.ridge(lamb)

            lambScore[i] = lamb, regressor.R2Score
        
        lambMax = lambScore[np.argmax(lambScore[:,1])]
        print("best Ridge:", lambMax)




         # OLS solution
        
        regressor = SGDR()
        regressor.craftDataset(N, p, None, None)
        regressor.craftX()
        regressor.OLS()
        print("OLS:", regressor.R2Score)

        if writeToFile:
            with open("parameterValues/SGDparameters_analytical.txt", "w") as outfile:
                lambScore = lambScore[lambScore[:,1].argsort()]
                outfile.write("Ridge, in order of worst to best:\n")
                for lamb, score in lambScore:
                    outfile.write(str(lamb) + " | " + str(score) + "\n")

                outfile.write("\nOLS | " + str(regressor.R2Score) + "\n")