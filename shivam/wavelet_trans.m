clear all

rawdata = dlmread('SPY.csv')


p_close(:,1) = rawdata(:,1)
p_close(:,2) = rawdata(:,5)
t = 0:1:length(p_close)