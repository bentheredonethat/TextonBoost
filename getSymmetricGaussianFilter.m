%{
getSymmetricGaussianFilter.m

ML founndations - 2012A
Yaniv Bar

Get a rotationally symmetric Gaussian lowpass filter 
of size 'filterSize' with standard deviation 'sigma' 

Input:
filterSize - filter size
sigma - standard deviation

Output:
gaussianFilter -  a rotationally symmetric Gaussian lowpass filter
%}

function [gaussianFilter]=getSymmetricGaussianFilter(filterSize, sigma)
gaussianFilter=myGaussianFilter(filterSize,sigma,'S');
% alt: gaussianFilter=fspecial('gaussian',filterSize,sigma);
