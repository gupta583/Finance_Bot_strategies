import pandas as pd
import numpy as np
import gdax
from datetime import datetime

def getTimeStamp(timeStart, timeEnd='today'):
    '''inputs:
    timeStart - start date in the format of mm/dd/yyyy ex: 5/1/2017
    timeEnd - start date in the format of mm/dd/yyyy ex: 10/1/2017
    timeEnd defaults to the start of the current day
    output - linux timestamp for start time and end time
    Exception - if timeEnd is earlier than timeStart the function will throw an Error!
    '''
    s = pd.Series([timeStart, timeEnd])
    dates = pd.to_datetime(s, infer_datetime_format=True)
    linuxTime = pd.DatetimeIndex(dates)
    index = linuxTime.astype(np.int64) // 10**9

    if index[1] - index[0] <= 0:
        myError = ValueError('timeEnd is less than timeStart !')
        raise myError
    return linuxTime



def getBoundedData(product_id, isoDates, timeInterval):
        '''Gets historic Rates between the itervals specified in isoDate
        inputs: product_id - the product identifier such as 'BTC-USD'
        isoDates - daterange object in ISO 8601 format
        timeInterval - GDAX REST API requires the frequency of sampling
        Look at the docs to find out more
        outputs: A List Lists each a inner list is a datapoint for
        our multivariate time series'''
        print(isoDates)
        client = gdax.PublicClient()
        return client.get_product_historic_rates(product_id, isoDates[0], isoDates[1], granularity=timeInterval)


def convToSeconds(timestep):
        """helper function for 'getHistoricalData'
        input timestep: same as input for the function below
        output int for seconds"""
        if timestep == 'monthly':
            return 60*60*24*30
        elif timestep == 'daily':
            return 60*60*24
        elif timestep == 'biweekly':
            return 60*60*24*14
        elif timestep == 'weekly':
            return 60*60*24*7
        elif timestep == 'min30':
            return 60*30
        elif timestep == 'hour6':
            return 60 * 60 * 6
        elif timestep == 'hourly':
            return 60*60
        else:
            if len(timestep) == 0:
                myError = ValueError('string is empty!')
                raise myError
            elif timestep[-1] == 's':
                timestep = timestep[:-1]
            else:
                myError = ValueError('String is not in the right form refer to doc string !')
            raise myError
        return int(timestep)


def getHistoricalData(product_id,dates,timestep = 'hourly'):
    """Retrive historical data
    Inputs: dates - list with 2 linuxtime stamps (required to use API)
    timestep - string that can be 'monthly' (30d), 'daily' (1d),
    'biweekly'(14d), 'weekly'(7d), 'min30' (30m) 'hour6' (6h)
    Above timesteps are the same as used by GDAX
    For user specific timestep use following form: 'seconds + s'
    Example: 100s (This will make a request to get data points every 100 seconds
    between the difference of the dates)
    - defaults to hourly data
    Outputs 2d array with each row containing time: OHLCV - (Open Hi Lo Close Volume)
    Exceptions: If the difference between dates is less than the user specified timestep
    function will throw an error!
    """

    # Need time step to be seconds
    timestepS = convToSeconds(timestep)
    # Calculate maximum range for API call
    numCalls = 200 * timestepS
    numCalls = numCalls.__str__()
    numCalls = numCalls + 'S'

    dateRange = pd.date_range(dates[0],dates[1],freq=numCalls)
    print(dateRange)
    #dateRange returns only 1 value if freq + date[0] > dates[1]
    if len(dateRange) == 1:
        tempDF = pd.DataFrame(index=dates)
    else:
        tempDF = pd.DataFrame(index=dateRange.union(dates))

    print(tempDF)
    # Convert to ISO 8601
    tempDF.index = tempDF.index.map(lambda x: datetime.strftime(x, '%Y-%m-%dT%H:%M:%S'))
    historicalList = []
    print(tempDF.index)
    for count in range(0,len(tempDF.index) - 1):
        dat = getBoundedData(product_id,tempDF.index[count:count+2],timestepS)
        for line in dat:
            if not isinstance(line,str):  # Occasionally the list returned contains strings
                historicalList.append(line)

    historicalDF = pd.DataFrame(historicalList)
    historicalDF.index = pd.to_datetime(historicalDF[0],unit='s')
    historicalDF.drop(historicalDF.columns[0],axis = 1,inplace = True)
    col = ['low', 'high', 'open', 'close', 'volume']
    historicalDF.columns = col
    # historicalDF.drop_duplicates()
    historicalDF.sort_index(axis = 0, inplace = True)
    historicalDF.index.name = 'time'
    return historicalDF

def createHistCSV(product_id,data,dateStart, dateEnd, timestep):
    dates = pd.to_datetime([dateStart,dateEnd])
    dates = dates.astype(str)
    dateStart = dates[0]
    dateEnd = dates[1]
    filename = product_id + ':' + dateStart + ':' +  dateEnd + ':'+ timestep.__str__()+'.csv'
    data.to_csv(filename)

def createData(product_id,dateStart,dateEnd,save=False,timestep='hourly'):
    dates = getTimeStamp(dateStart,dateEnd)

    data = getHistoricalData(product_id, dates, timestep=timestep)
    if save:
        createHistCSV(product_id,data, dateStart = dateStart, dateEnd = dateEnd, timestep = timestep)
    return data

