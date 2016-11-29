%{
calcFeatureResponse.m 

ML founndations - 2012A
Yaniv Bar

Calculate feature response given rectangle tl and br positions
and texton integral image.

Input:
r_tl - rect tl point
r_br - rect br point
textonIntIm - texton integral image

Output:
featureResponse - feature response value
featureResponsePerc - feature response perc
%}

function [featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm)
% s1 s2
% s3 s4
r_tl_x=r_tl(1);
r_tl_y=r_tl(2);
r_br_x=r_br(1);
r_br_y=r_br(2);
areaDim=size(textonIntIm(r_tl_y:r_br_y,r_tl_x:r_br_x));
area=areaDim(1)*areaDim(2);
s4=textonIntIm(r_br_y,r_br_x);
s2=textonIntIm(r_tl_y,r_br_x);
s3=textonIntIm(r_br_y,r_tl_x);
s1=textonIntIm(r_tl_y,r_tl_x);
featureResponse = s4-s3-s2+s1;
featureResponsePerc = (featureResponse / area);
