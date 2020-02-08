from sklearn.model_selection import train_test_split
from Util import dwt
from Util import finUtil
import numpy as np

class PreProcessor:
    def __init__(self,data,a,b):
        self.__data = data
        self.__scaler = None
        self.__processed = self.__data.copy(deep = True)
        self.minmax = None
        self.minfactor = a
        self.maxfactor = b
        self.featureRange = None
        self.initialCond = None
    def __myMinMaxTransform(self,data):
        name = data.name
        mindat = self.minmax[name][0]
        maxdat = self.minmax[name][1]
        a = self.minfactor
        b = self.maxfactor
        return (data.values - mindat*a) / (maxdat*b - mindat*a)
    def __myMinMaxInverse(self,data):
        name = data.name
        mindat = self.minmax[name][0]
        maxdat = self.minmax[name][1]
        a = self.minfactor
        b = self.maxfactor
        return data * (maxdat*b - mindat*a) + mindat*a
    # def __myDifferenceTransform(self,data):

    def normalize(self,data,featureRange = (0,1),type='minmax',names = None):
        self.featureRange = featureRange
        if names is None: names = data.columns
        if type == 'difference':
            data[names] = data[names].diff()

        if self.minmax is None:
            allnames = data.columns
            maxval = data.max()
            minval = data.min()
            minmax = np.array([minval.values,maxval.values]).T
            self.minmax = {}
            for k in range(len(allnames)):
                self.minmax[allnames[k]] = minmax[k]

        data[data.columns] = data.apply(self.__myMinMaxTransform)
        data[data.columns] = data*(featureRange[1] - featureRange[0]) + featureRange[0]
        return data

    def normalizeInverse(self,data,initial, type='minmax',names=None):

        if self.minmax is None:
            print('Error normalize must be called first before taking inverse')
            return
        temp = (data - self.featureRange[0])/(self.featureRange[1] - self.featureRange[0])
        ret =  temp.apply(self.__myMinMaxInverse)

        if type == 'difference':
            ret.iloc[0] = initial
            ret[names] = ret[names].cumsum()
            return ret
        else: return ret



    def split(self,splitPer,shuffle = False):
        train, test = train_test_split(self.__processed,train_size=splitPer,shuffle=shuffle)
        return train , test
    def denoise(self,denoiseNum):
        temp = self.__processed[self.__processed.columns[0:-1]]
        for k in range(denoiseNum):
            temp = temp.apply(dwt.denoise)
        self.__processed[self.__processed.columns[0:-1]] = temp
    def dropNA(self):
        self.__processed.dropna(inplace=True)
    def generate(self,drop = False):
        self.denoise(denoiseNum=2)
        finUtil.addAllIndicators(self.__processed,True)
        if drop is True:
            self.__processed.dropna(inplace=True)
    def getData(self,original = False, isReference=True):
        if original and isReference:
            return self.__data
        elif not original and isReference:
            return self.__processed
        elif original and not isReference:
            return self.__data.copy()
        else:
            return self.__processed.copy()
    def getScaler(self):
        return self.__scaler

    def createWindow(self,data, look_backX=10, look_backY=1,names = ['close']):
        num_rows = len(data)
        x_data = []
        y_data = []
        i = 0
        while ((i + look_backX + look_backY) <= num_rows):
            x_window_data = data[i:(i + look_backX)]
            y_window_data = data[(i + look_backX):(i + look_backX + look_backY)]
            x_data.append(x_window_data.values)
            y_data.append(y_window_data[names].values)
            i = i + 1
        dim = np.shape(y_data)[2]
        return [np.array(x_data), np.array(y_data).reshape((len(y_data),dim))]

