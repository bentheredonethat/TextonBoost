%{ 
testIntImg.m

ML founndations - 2012A
Yaniv Bar

%}
clear;

DEBUG_FLAG=0;
imgFileName = 'textonizationTestImg.bmp';

if(DEBUG_FLAG==1)
    I=[1,1,1;1,1,1;1,1,1];
else
    inp=imgFileName;
   
    L=256;
    I=imread(inp);
    [I_n,I_m,I_b]=size(I);
    if(I_b==3)
        I=rgb2gray(I);
    end
    I=cast(I,'double')*(1/(L-1));
end
intImg_I = calcIntImg(I);

if(DEBUG_FLAG==1)
    I
    intImg_I
else
    figure;
    subplot(1,2,1);imshow(I);
    subplot(1,2,2);imagesc(intImg_I);
end





