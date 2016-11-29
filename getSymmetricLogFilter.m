%{
getSymmetricLogFilter.m

ML founndations - 2012A
Yaniv Bar

Get a rotationally symmetric LOG filter 
of size 'filterSize' with standard deviation 'sigma' using fspecial method

Input:
filterSize - filter size
sigma - standard deviation

Output:
logFilter -  a rotationally symmetric LOG filter
%}

function [logFilter]=getSymmetricLogFilter(filterSize, sigma)
logFilter=fspecial('log',filterSize,sigma);


