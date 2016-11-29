%{
myGaussianFilter.m

ML founndations - 2012A
Yaniv Bar

Input:
filterSize - filter size
filterStd - filter standard deviation
der - (X) 1D-X, (Y) 1D-Y, (S) 2D - Sym (X & Y)

Output:
outputFilter - filter

%}
function [outputFilter]=myGaussianFilter(filterSize,filterStd, der);
filter_n = filterSize(1);
filter_m =  filterSize(2);

[x,y] = meshgrid(-floor(filter_m/2):floor((filter_m-1)/2),-floor(filter_n/2):floor((filter_n-1)/2));
% generating gaussian filter

if(strcmp(der,'S')==1)
    D=sqrt(x.^2+y.^2);
    outputFilter=exp(-(D.^2)./(2*(filterStd.^2)));
elseif(strcmp(der,'X')==1)
    D=sqrt(x.^2);
    outputFilter=exp(-(D.^2)./(2*(filterStd.^2)));
elseif(strcmp(der,'Y')==1)
    D=sqrt(y.^2);
    outputFilter=exp(-(D.^2)./(2*(filterStd.^2)));
end

outputFilter=outputFilter./(sum(sum(outputFilter)));

