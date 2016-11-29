%{
testSubSampleImg.m

ML founndations - 2012A
Yaniv Bar

%}
clear;

imgFileName = 'textonizationTestImg.bmp';
Im = imread(imgFileName);
Im = im2double(Im);
[Im_n,Im_m,Im_d]=size(Im);
if(Im_d ~= 3)
    error('Only work on colored images!')
    exit;
end

ImOut = subSampleImg(Im,[5,5]);

figure;
subplot(1,2,1);imshow(Im);
subplot(1,2,2);imshow(ImOut);





