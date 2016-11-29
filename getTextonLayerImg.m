%{
getTextonLayerImg.m

ML founndations - 2012A
Yaniv Bar

Get the texton layer image based on textonized image and a texton value.

Input:
t - texton
textonizedIm - textonized image

Output:
textonLayerIm - texton layer image
%}
function [textonLayerIm]=getTextonLayerImg(t,textonizedIm)
[n,m]=size(textonizedIm);
fullTextonLayerIm=ones(n,m).*t;
textonLayerIm=(textonizedIm==fullTextonLayerIm);
 

