%{
calcBoundingBox.m

ML founndations - 2012A
Yaniv Bar

Calculate bounding box given a pixel and image dimension.
Generally the bounding box is  covering
about half the image area. The bounding box was
±100 pixels in x and y when possible.

Input:
i - a pixel coordinate
im_dim - image dimensions
FIXED_BOUNDING_BOX_FLAG - 0/1 (0 - relative size, 1 - fixed size)

Output:
boundingBox_tl - tl point
boundingBox_br - br point
%}
function [boundingBox_tl, boundingBox_br] = calcBoundingBox(i,im_dim,FIXED_BOUNDING_BOX_FLAG)
i_X=i(1);
i_Y=i(2);
im_n=im_dim(1);
im_m=im_dim(2);

if(FIXED_BOUNDING_BOX_FLAG==0)
    boundingBox_X=floor(im_m/8);
    boundingBox_Y=floor(im_n/8);
end
if(FIXED_BOUNDING_BOX_FLAG==1)
    boundingBox_X=10;
    boundingBox_Y=10;
end

if(i_X-boundingBox_X<1)
    boundingBox_tl(1)=1;
else
    boundingBox_tl(1)=i_X-boundingBox_X;
end

if(i_X+boundingBox_X>im_m)
    boundingBox_br(1)=im_m;
else
    boundingBox_br(1)=i_X+boundingBox_X;
end

if(i_Y-boundingBox_Y<1)
    boundingBox_tl(2)=1;
else
    boundingBox_tl(2)=i_Y-boundingBox_Y;
end

if(i_Y+boundingBox_Y>im_n)
    boundingBox_br(2)=im_n;
else
    boundingBox_br(2)=i_Y+boundingBox_Y;
end





    
    
