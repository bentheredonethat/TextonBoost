%{
subSampleImg.m

ML founndations - 2012A
Yaniv Bar

Subsampling an image given mask size. 
this is done by taking the center pixel out of each mask.

Input:
f - input image 
maskSize - mask size

Output:
g - reduced output image

%}

function [g]=subSampleImg(f,maskSize)
[f_n,f_m,f_d]=size(f);
mask_n=maskSize(1);
mask_m=maskSize(2);

if(mod(mask_n,2)==0 || mod(mask_m,2)==0)
    display('Error! mask dim must be of odd size');
    exit;
end

indices_n=[1:mask_n:f_n];
indices_m=[1:mask_m:f_m];
g=f(indices_n,indices_m,:);




