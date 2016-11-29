%{
getFilterBank.m

ML founndations - 2012A
Yaniv Bar

Get the following 11 - dimensional filter bank.
The filter bank is made of 3 Gaussians, 4 Laplacian of Gaussians (LoG) and 4 first order derivatives of Gaussians.
(three Gaussian kernels (with sigma = 1, 2, 4) , four LoGs kerneks (with sigma = 1, 2, 4, 8), four derivatives of Gaussians (x and y aligned sets), 
each with two different values of kernels (sigma = 2, 4). 

Input:
filterSize - filter size

Output:
filtrerBank - 11 dimensional filter bank (cell)
%}
function [filtrerBank]=getFilterBank(filterSize)
DISP_FILTER_BANK_FLAG=0;
type1_numFilters = 3;
type2_numFilters = 4;
type3_numFilters = 4;

sigma_val1=1;
sigma_val2=2;
sigma_val4=4;
sigma_val8=8;

gaussianFilter = getSymmetricGaussianFilter(filterSize,sigma_val1);
cell{1,1}=gaussianFilter;
gaussianFilter = getSymmetricGaussianFilter(filterSize,sigma_val2);
cell{1,2}=gaussianFilter;
gaussianFilter = getSymmetricGaussianFilter(filterSize,sigma_val4);
cell{1,3}=gaussianFilter;
logFilter = getSymmetricLogFilter(filterSize,sigma_val1);
cell{1,4}=logFilter;
logFilter = getSymmetricLogFilter(filterSize,sigma_val2);
cell{1,5}=logFilter;
logFilter = getSymmetricLogFilter(filterSize,sigma_val4);
cell{1,6}=logFilter;
logFilter = getSymmetricLogFilter(filterSize,sigma_val8);
cell{1,7}=logFilter;
gaussianFilter = getOneDimGaussianFilter(filterSize,sigma_val2,'X');
cell{1,8}=gaussianFilter;
gaussianFilter = getOneDimGaussianFilter(filterSize,sigma_val4,'X');
cell{1,9}=gaussianFilter;
gaussianFilter = getOneDimGaussianFilter(filterSize,sigma_val2,'Y');
cell{1,10}=gaussianFilter;
gaussianFilter = getOneDimGaussianFilter(filterSize,sigma_val4,'Y');
cell{1,11}=gaussianFilter;

filtrerBank = cell;

if(DISP_FILTER_BANK_FLAG == 1)
    figure;
    for i=1:type1_numFilters
        subplot(2,type1_numFilters,i); imagesc(filtrerBank{i});colormap(gray);
        subplot(2,type1_numFilters,i+type1_numFilters); mesh(filtrerBank{i});
    end
    figure;
    for i=1:type2_numFilters
        subplot(2,type2_numFilters,i); imagesc(filtrerBank{type1_numFilters+i});colormap(gray);
        subplot(2,type2_numFilters,i+type2_numFilters); mesh(filtrerBank{type1_numFilters+i});
    end
    figure;
    for i=1:type3_numFilters
        subplot(2,type3_numFilters,i); imagesc(filtrerBank{type1_numFilters+type2_numFilters+i});colormap(gray);
        subplot(2,type3_numFilters,i+type3_numFilters); mesh(filtrerBank{type1_numFilters+type2_numFilters+i});
    end
end
