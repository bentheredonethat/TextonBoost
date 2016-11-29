%{
getOneDimGaussianFilter.m

ML founndations - 2012A
Yaniv Bar

Get a derivative Gaussian lowpass filter 
of size 'filterSize' with standard deviation 'sigma' 

Input:
filterSize - filter size
sigma - standard deviation
der - (X) 1D-X, (Y) 1D-Y

Output:
gaussianFilter -  a one directional Gaussian lowpass filter
%}

function [gaussianFilter]=getOneDimGaussianFilter(filterSize, sigma, der)
if(strcmp(der,'X')==1)
  gaussianFilter=myGaussianFilter(filterSize,sigma,'X');
elseif(strcmp(der,'Y')==1)
  gaussianFilter=myGaussianFilter(filterSize,sigma,'Y');
end