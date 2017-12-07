# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:21:55 2017

@author: sandeep
"""

'''
This file contains two classes:
1. PhotonAnalyser takes photon numbers detected in a certain number of subbins
and calulates a probability that the detection was initially bright or dark.

2. StateAnalyser takes the data from PhotonAnalyser and calclate the most likely 
projection on to the z-axis and error bars are calculated from posterior probability
distribution.

I have set the mean dark and bright counts to some nominal levels, you can use the
PhotonAnalyser.calculateCountRates() function to try and fit the distribution but 
I didn't automate this in case the fit does not converge.

There is an usage example at the bottom.
'''

import numpy as np
import matplotlib.pyplot as plt
import progressbar as pb
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import factorial, erf


decayT_B = 3.786e-3     # in seconds, calculated from fitting calibration data
decayT_D = 49.6e-3

photonR_B = 40e3    #photons / second
photonR_D = 430 #See bottom decay calibration.py comes from optical pumping expt.

subbinT = 160e-6     # in seconds

maxPhotons = 30    # maximum photons per subbin up to which to evaluate the probability mass functions

#print 'imported subbin'

class PhotonAnalyser:
    ''' 
    Class which analyses subbined photon data. The method is identical to that in 
    State selective detection of hyperfine qubits
    http://dx.doi.org/10.1088/0953-4075/48/7/075101
    
    The parameters that I required to make this work are the photon count rates in the 
    bright and dark states and the decay rate between the two states.
    '''   

    def __init__(self, numBins=5, subbinT=subbinT, photonR_B=photonR_B, 
                         photonR_D=photonR_D, maxPhotons=50, decayT_B=decayT_B,
                         decayT_D=decayT_D,filename=None, fullLocation=None, 
                         photonArray=None, coolingArray=None, coolingThreshold=5):
        '''
        Initialise instance of class, require some basic information.
        Input: 
            numBins = number of subbins used in the data collection
            subbinT = detection time for each subbin (total detection time = numbins * subbinT). Units = seconds
            photonR_B = The average photon count rate in the bright state. Units = photons/second
            photonR_D = The average photon count rate in the dark state. Units = photons/second
            decayT_B = The decay lifetime from the bright state to the dark state. Units = seconds
            decayT_D = The decay lifetime from the dark state to the bright state. Units = seconds
            maxPhotons = the maximum number of photons to evaluate the liklihood distributions.
            photonArray = These are the detected photons that you want to analyse
                        Must be x number of rows and numBins*r columns.
            coolingArray = a set of photons detected during the initial cooling period, mean count number 
                        usually roughly the same as 800 uS detection period. If this array doesn't exist then
                        assumes that all data is good.
            coolingThreshold = minimum number of photons that must be detected during the cooling period to say
                        that the ion was present and in a good state for experiments.
        '''
                             
        self.numBins = numBins
        self.subbinT = subbinT
        self.photonR_B = photonR_B
        self.photonR_D = photonR_D
        self.maxPhotons = maxPhotons
        self.decayT_B = decayT_B
        self.decayT_D = decayT_D
        
        
        if fullLocation:
            self.photonArray = self.photonDataLoad(fileLocation=fullLocation)
        
        elif filename: 
            folderLocation='X:/Ytterbium (Sim Trap)/Randomised Benchmarking/Data/'
            fullLocation = folderLocation + filename[:-3] + '/' + filename + '/' + filename + 'photons.txt'
            self.photonArray = self.photonDataLoad(fileLocation=fullLocation)
        
        elif photonArray.any():
            self.photonArray = photonArray.astype(np.int16)
            
        else:
            return 'No valid data.'
        
        dim = self.photonArray.shape
        #print 'in initial function'    
        #print coolingArray
        #print coolingArray.sum()
        #print int(coolingArray.sum()) == 0
        
        if coolingArray == None:
            self.coolingThreshArray = np.ones([dim[0],dim[1]/self.numBins], dtype=bool)
            
        
        elif int(coolingArray.sum()) == 0:
            self.coolingThreshArray = np.ones([dim[0],dim[1]/self.numBins], dtype=bool)
            #print self.coolingThreshArray
        else:
            self.coolingThreshArray = np.greater_equal(coolingArray, coolingThreshold)
            print(self.coolingThreshArray.shape)
        
    def calculateCountRates(self):
        
        photonCounts = self.photonArray        
        
        dim = photonCounts.shape
        
        countRates = photonCounts.reshape(-1,dim[1]/self.numBins,self.numBins).sum(axis=2)
        countRates = countRates.reshape(countRates.size)
        
        self.histogram_x = np.amax(countRates)+1
        
        self.photonHist = np.histogram(countRates, bins=np.arange(self.histogram_x))
        
        brightCounPerDetect = self.photonR_B * (self.numBins * subbinT)
        darkCountPerDetect = self.photonR_D * (self.numBins * subbinT)
        
        #inital guesses for the fitting proceedure
        initial = [darkCountPerDetect, brightCounPerDetect, dim[1]/self.numBins, dim[1]/self.numBins,10]
        
        #print 'initial', initial
        #print 'Dark mean, bright mean, dark amplitude, bright amplitude, central plateau level'        
        
        popt, pcov = curve_fit(self.doublePoisson, self.photonHist[1][:-1], self.photonHist[0], p0=initial)        
        
        #print popt
        
        #x_fit = np.linspace(0,60,1e3)        
        
        #plt.semilogy(self.photonHist[1][:-1],self.photonHist[0])
        #plt.semilogy(x_fit, self.doublePoisson(x_fit,*popt))
        #plt.show()
        
        self.photonR_B = popt[1] / (self.numBins * self.subbinT)
        
        return popt
        
        
    
        
        
    def doublePoisson(self, x, a, b, c, d, e):
        
        return c * a**x * np.exp(-a) / factorial(x) + d * b**x * np.exp(-b) / factorial(x) + e * ( (erf(x-a) - 0.5) * ( 0.5- erf(x-b) ) )
        
        
        
        
    def fac(self, n):
        '''
        Calculates the factorial of some number, n, recursively. Used to calculate 
        Poisson distributions.
        '''
        if n == 0 or n == 1:
            return 1
        else:
            return n*self.fac(n-1)
        
        
        
        
        
        
        
    def calculateDecayDistributions(self): 
        '''
        Calculates the liklihood function for the cases where the ion changes 
        state from bright to dark or visa versa. This is done by calculating the 
        overlap integral of a weighting function equations 9 and 10 in the paper 
        and the photon count rate at the different mean photon counts.
        
        Output: XBD = array containing the likelihood of detecting a certain number of 
                        photons if the ion decays from bright to dark
                XDB = array containing the likelihood of detecting a certain number of 
                        photons if the ion decays from dark to bright
        '''
        
        
        integrationStepSize = 1e-2  #This is the resolution of the integration.
        meanPhoton = np.arange(0,self.maxPhotons,integrationStepSize)
        
        #called g_BD in paper, equation 9
        g_BD = np.exp(-(meanPhoton - self.photonR_D * self.subbinT)/(self.photonR_B * self.decayT_B ) ) / (self.photonR_B*self.decayT_B)
        
        #called g_DB in paper, equation 10
        g_DB = np.exp(-( (self.photonR_B + self.photonR_D) * self.subbinT - meanPhoton) \
        / ( self.photonR_B * self.decayT_D ) ) / (self.photonR_B * self.decayT_D)
        
        # integrate weightfunction (g_BD or g_DB) mulitplied by poission distribution for each possible photon count   
        
        
        
        self.XBD = np.zeros(self.maxPhotons)
        self.XDB = np.zeros(self.maxPhotons)
        
        #The integration range is between the photon count rates for the bright and darks states.
        int_range = np.logical_and(meanPhoton >= self.photonR_D*self.subbinT, 
                                   meanPhoton <= (self.photonR_B+self.photonR_D)*self.subbinT)
        poisson = lambda n: np.exp(-meanPhoton[int_range]) * meanPhoton[int_range]**n / factorial(n)
        
        #Calculate the integral of poisson distributions multiplied by the weighting factor, equation 6 in paper
        for photon_number in range(maxPhotons):
            
            self.XBD[photon_number] = np.trapz(g_BD[int_range] *   
                    poisson(photon_number), meanPhoton[int_range])
            self.XDB[photon_number] = np.trapz(g_DB[int_range] *   
                    poisson(photon_number), meanPhoton[int_range])
      
        return self.XBD, self.XDB


        
        
        
        
    def calculateSteadyStateDistributions(self): 
        '''
        Calculates the photon distributions for the cases where the ion remains in 
        either bright or dark state.
        
        Output:
            brightDistribution = array containing probability mass function for a Poisson 
            distribution with mean rate (self.photonR_B+self.photonR_D)*self.subbinT
            darkDistribution = array containing probability mass function for a Poisson 
            distribution with mean rate self.photonR_D*self.subbinT
        '''

        
        
        photonRange = np.arange(0,self.maxPhotons)
        
        self.Dist_B = np.exp(-self.subbinT/self.decayT_B) * \
            stats.poisson.pmf( photonRange, (self.photonR_B+self.photonR_D)*self.subbinT )
        
        self.Dist_D = np.exp(-self.subbinT/self.decayT_D) * \
            stats.poisson.pmf( photonRange, self.photonR_D*self.subbinT )
        
        return self.Dist_B, self.Dist_D
        
        
        
        
        
        
    def calculateStateLikelihood(self): 

        ''' 
        This function calculates the most likely state for a particular set of detected photons.
        Input:
            photonArray = These are the detected photons that you want to analyse
                        Must be x number of rows and numBins*r columns
                        
        Output:
            An 3d array containing the likelihood of being in the dark state and the bright 
            state. First dimension is only 2 long containing the likelihoods (bright first).
            Dimensions x number of rows and r columns and two items in 3rd dimension
            
                
        '''
                
        
        #Input array size 
        #dim[0] is number of row, sequences in RB measurements
        #dim[1] is number of columns, numBins * noise realisations
        dim = self.photonArray.shape
        
        
        if len(dim) == 1:
            self.photonArray = self.photonArray[np.newaxis, :]
            dim = tuple((1, dim[0]))
            
            
        #Check if the array is of the correct shape
        if np.mod(dim[1],self.numBins): 
            print 'Incorrect array dimensions'
            return
        
        #Initalise return matrix
        #number of rows is number of experiments
        outputLikelihood = np.zeros([dim[0], dim[1]/self.numBins, 2]) 
        stateArray = np.zeros([dim[0], dim[1]/self.numBins])


        
        for row in range(dim[0]):
            #calculate the number of valid reps given the thresholded cooling data
            #Create an array in each element of outputLikelihood with this length
            
            
            for rep in range(dim[1]/self.numBins):
                
                if self.coolingThreshArray[row, rep]:                
                
                    likelihood = np.matrix([ [1,0], [0,1] ])    #Initialise likelihood matrix as identity 
                    
                    
                    for i in range(self.numBins):
                        
                        subbinLikelihood = np.matrix([ [self.Dist_B[self.photonArray[row,rep*self.numBins+i]], 
                                                        self.XDB[self.photonArray[row,rep*self.numBins+i]] ], 
                                                        [self.XBD[self.photonArray[row,rep*self.numBins+i]] ,
                                                         self.Dist_D[self.photonArray[row,rep*self.numBins+i]] ] ])
                        #print 'subbin'
                        #print subbinLikelihood
                        likelihood = np.dot(subbinLikelihood,likelihood)
                        #print 'cumalative likelihood'
                        #print likelihood
                        #probability of being in the bright state
                        outputLikelihood[row,rep,0] = np.sum(likelihood[:,0])
                        #probability of being in the dark state
                        outputLikelihood[row,rep,1] = np.sum(likelihood[:,1])
                        
                        
                        '''
                        print 'subbin liklihood'
                        print subbinLikelihood
                        print 'full likelihood'
                        print likelihood
                        '''
    
                    #normalise the likelihoods such that Pup + Pdown = 1
                    
                    normalisation = outputLikelihood[row,rep,0] + outputLikelihood[row,rep,1]
                    #likelihold of initally being bright
                    outputLikelihood[row,rep,0] = outputLikelihood[row,rep,0] / normalisation
                    #likelihood of initally being dark
                    outputLikelihood[row,rep,1] = outputLikelihood[row,rep,1] / normalisation
                    
                    stateArray[row, rep] = np.greater(outputLikelihood[row, rep, 0], outputLikelihood[row, rep, 1])                     


                else:
                    outputLikelihood[row,rep,:] = -1
                    stateArray[row, rep] = -1
                    
                        
                    
        return stateArray, outputLikelihood
            
    
     
    def photonDataLoad(self, fileLocation):
        
        print fileLocation
        
        photonData = np.loadtxt(fileLocation)
        
        return photonData
               
        
        
        
    def evaluatePhotons(self):
        '''
        Bringing together the earlier functions and outputting a single matrix with the states 
        for each repetition.
        Output:
            stateArray = This is the most likely state for the collected photons 
                        False = Dark, True = Bright
                        dimensions = x rows and rep numbers of columns
        '''
        
        #calculate the distributions for the different for either remaining in the same state or 
        #changing state throughout the detection.
        self.calculateDecayDistributions()
        self.calculateSteadyStateDistributions()
            
        
        return  self.calculateStateLikelihood()

  





##########################################################################################################################################################



class StateAnalyser:
    '''    
    Take likelihoods for both the bright and dark state for multiple repetitions
    of the same experiment. Then use this to calculate to projection on the z-axis.
    '''
    
    
    def __init__(self, likelihoodArray, stateResolution = int(1e3)): 
        '''Set the resolution to which we test the z-projection in the init function
            likelihoodArray comes from the PhotonAnalyser class
        '''
        self.stateResolution = stateResolution
        self.likelihoodArray = likelihoodArray
    
    def projectionProb(self, realState):
       ''' Simple function which calculates 
       Tr( E(up) realState ) and Tr( E(down) realState ). '''
       
       upProb = realState
       downProb = 1 - realState

       return upProb, downProb
       
     
     
    def stateEvaluate(self):
    
         '''
         For each individual sequence (rows), calculate the likelihood of of each possible
         z-state, resolution is set in the init function. Add extra information for each 
         new rep. Apply a Bayesian technique where the prior is updated each time.
         '''     
         
         dim = self.likelihoodArray.shape
         #print dim
         #Input array size 
         #dim[0] is number of row, sequences x noise realisations in RB measurements
         #dim[1] is number of columns, reps

         #Projection array contains the likelihood for each seqeunce and noise realisation 
         # as a function of possible state
         self.projectionArray = np.zeros((dim[0], self.stateResolution))  
         
         
         
         #Maximum likelihood is the weighted mean of projection array to say what is 
         #our most likely z-projection.
         self.maximumLikelihood = np.zeros(dim[0])
         
         errorBars = np.zeros((2,dim[0]))
         #upperBar = np.zeros(dim[0])
         
         #put a progrsss bar in
         l = dim[0] 
         pB = pb.ProgressBar(l, 'Progress')
        
         #Increment over rows of data, this can just be data points or sequences and noise
         #realisations
        
         downProb = np.zeros(self.stateResolution)
         upProb = np.zeros(self.stateResolution)         
         
         for i in range(self.stateResolution):
             
             upProb[i], downProb[i] = self.projectionProb(float(i)/self.stateResolution)
             
             
         
         for expts in range(dim[0]):
             #print expts
             #stateLikelihood is the likelihood for a particular z-projection for
             #an individual experiment 
             
             #if np.mod(expts,10) == 0:
                 
                 #print 'Calculating point', expts
             
             self.stateLikelihood = np.ones(self.stateResolution)
             
             #number of reps for this expt
             
             #Step over possible z-projections and calculate their likelihoods.
             
             
             for rep in range(dim[1]):
                 
                 for stateStep in range(self.stateResolution):
                 
                     if int(self.likelihoodArray[expts, rep, 0]) is not -1:
                     
                         #Update the posterior with every valid repetition
                         self.stateLikelihood[stateStep] = self.stateLikelihood[stateStep] * ( self.likelihoodArray[expts, rep, 1] * downProb[stateStep] + self.likelihoodArray[expts, rep, 0] * upProb[stateStep])
                 '''        
                 if rep < 20:
                     print 'Repetition ' + str(rep)
                     print self.stateLikelihood[stateStep]
                     #print self.likelihoodArray[expts, rep, 0], self.likelihoodArray[expts, rep, 1]
                     #print downProb, upProb
                     
                     print self.stateLikelihood
                 '''
                 #normalisation of the likelihood such that the integral is one
                 self.stateLikelihood = self.stateLikelihood / np.sum(self.stateLikelihood)
             
             self.projectionArray[expts,:] = self.stateLikelihood
             
             #weighted mean to find the most likely state.
             self.maximumLikelihood[expts] = np.dot(self.projectionArray[expts,:],range(self.stateResolution)) / self.stateResolution
             
             #Try and calculate the error bars
             
             cumulative_prob = np.cumsum(self.projectionArray[expts,:])
             confidence_interval = 0.682
             
             #lower errorbar
             errorBars[0,expts] = self.maximumLikelihood[expts] - float(interp_errorbar((1 - confidence_interval)/2, cumulative_prob)) / self.stateResolution
             
             #upper errorbar             
             errorBars[1,expts] = float(interp_errorbar((1 + confidence_interval)/2, cumulative_prob)) / self.stateResolution - self.maximumLikelihood[expts]
             
             pB.update()
             
             
             
         return self.maximumLikelihood, errorBars
         
         
     
    
def find_nearest(value, array):
    return np.argmin(np.abs(array - value))
    
    
def interp_errorbar(value, array):
    closest = np.argmin(np.abs(array - value))
    
    if array[closest] < value:
        lower = closest
        upper = lower+1
    else:
        upper = closest
        lower = upper - 1
        
    grad = array[upper] - array[lower]
    interpolate = (value - array[lower])/grad

    return lower + interpolate
    
    
    
if __name__ == '__main__' and 1:
    
    #import time
    #linux path
    #photonNumbers = np.loadtxt('/home/sandeep/Documents/intrinsic noise/20160912002photons.txt')
    #coolingNumbers = np.loadtxt('/home/sandeep/Documents/intrinsic noise/20160912002cooling_photons.txt')
    
    #photonNumbers = np.loadtxt('/home/sandeep/Dropbox/rb paper data/test/20161102003photons.txt')
    #coolingPhotons = np.loadtxt('/home/sandeep/Dropbox/rb paper data/test/20161102003cooling_photons003.txt')    
    
    photonNumbers = np.loadtxt('C:/Users/sandeep/Dropbox/rb paper data/test/20161102003photons.txt')
    coolingPhotons = np.loadtxt('C:/Users/sandeep/Dropbox/rb paper data/test/20161102003cooling_photons003.txt')      
    
    pA = PhotonAnalyser(photonArray=photonNumbers, coolingArray=coolingPhotons)
    
    pA.calculateCountRates()
    states, likelihoods = pA.evaluatePhotons()
    
    sA = StateAnalyser(likelihoods)
    stateEstimate, stateErrors = sA.stateEvaluate()
    
    '''
    start = time.time()    
    
    #photonNumbers = np.loadtxt('/home/sandeep/Documents/intrinsic noise/intrinsic/20161031017/20161031017photons.txt')
    #coolingNumbers = np.loadtxt('/home/sandeep/Documents/intrinsic noise/intrinsic/20161031017/20161031017cooling_photons.txt')
    
    #windows path
    photonData = np.loadtxt('C:/Users/Sandeep/Documents/gst subbin/20161201019photons.txt')
    coolingData = np.loadtxt('C:/Users/Sandeep/Documents/gst subbin/20161201019cooling_photons019.txt')
    
    photonNumbers = {}
    
    photonNumbers['baseline'] = [photonData[0:-1:12],coolingData[0:-1:12]]
    photonNumbers['DC75Hz'] = [photonData[1:-1:12],coolingData[1:-1:12]]
    photonNumbers['DC500Hz'] = [photonData[2:-1:12],coolingData[1:-1:12]]  
    photonNumbers['DC1kHz'] = [photonData[3:-1:12],coolingData[1:-1:12]]
    photonNumbers['DC-1kHz'] = [photonData[4:-1:12],coolingData[1:-1:12]]  
    photonNumbers['DC1p4kHz'] = [photonData[5:-1:12],coolingData[1:-1:12]]   
    photonNumbers['slowFM25Hz'] = [photonData[6:-1:12],coolingData[1:-1:12]]   
    photonNumbers['bad_pumping'] = [photonData[7:-1:12],coolingData[1:-1:12]]   
    photonNumbers['50HzFM_5Hz'] = [photonData[8:-1:12],coolingData[8:-1:12]]   
    photonNumbers['50HzFM_20Hz'] = [photonData[9:-1:12],coolingData[9:-1:12]]    
    photonNumbers['slow_AM_drift_1pct'] = [photonData[10:-1:12],coolingData[10:-1:12]]    
    photonNumbers['AM_offset_1pct'] = [photonData[11:-1:12],coolingData[11:-1:12]]
    
    states = {}
    likelihoods = {}
    totalStates = {}


    for key in photonNumbers:
        
        print 'Calculating states for', key
        
        start = time.time()    
        pA = PhotonAnalyser(photonArray=photonNumbers[key][0], coolingArray=photonNumbers[key][1])
        pA.calculateCountRates()
        states[key], likelihoods[key] = pA.evaluatePhotons()
        
        numOfBright = (states[key] == 1).sum(axis=1)
        totalReps = (states[key] == 1).sum(axis=1) + (states[key] == 0).sum(axis=1)
        
        totalStates[key] = [numOfBright, totalReps]
        
        #sA = StateAnalyser(likelihoods)
        #stateEstimate, stateErrors = sA.stateEvaluate()
        
        end = time.time()    
        print 'Calculation time =',end-start
    #summedData = photonNumbers.reshape(-1,photonNumbers.shape[1]/5,5).sum(axis=2)
    '''
    #threshold = summedData < 3
    #thresholdData = np.mean(threshold,axis=1)
    
    
    #corneliusSubbin = np.loadtxt('/home/sandeep/Documents/subbin/Detection_II_flops/Detection_II/20160912002scan_settings/data_subbin_current.csv')
    #corneliusSubbin2 = np.loadtxt('/home/sandeep/Documents/subbin/Detection_II_flops/Detection_II/20160912002scan_settings/data_subbin_10000.csv')
    
    #x = range(stateEstimate.shape[0])        
    
    #plt.figure(figsize=(20,15))
    #plt.errorbar(x, stateEstimate, yerr = stateErrors, fmt ='o-', label = 'Maximum likelihood')
    #plt.plot(1-thresholdData, 'x-', label = 'Threshold')
    #plt.plot(corneliusSubbin, '*-', label = 'Cornelius current')
    #plt.plot(corneliusSubbin2, '*-', label = 'Cornelius 10000')
    #plt.legend()
    #plt.savefig('masked data.pdf')
    #plt.show()
    #plt.close()
    
    
    #dataOutput = np.genfromtxt('C:/Users/Sandeep/Documents/gst subbin/MyDataTemplate.txt', delimiter='  ', dtype = str)
    #np.savetxt('dataOutput.txt', dataOutput, fmt='%s', delimiter = '  ', header='## Columns = plus count, count total')