%{
testImgTextonization.m

ML founndations - 2012A
Yaniv Bar

%}

clear;

imgFileName = 'textonizationTestImg.bmp';
%imgFileName = 'textonizationTestImg2.bmp';
%imgFileName = 'textonizationTestImg3.bmp';
%imgFileName = 'textonizationTestImg4.bmp';
outImgFileName = sprintf('%s_TM.bmp', imgFileName(1:end-4));
outImgFileNameForView = sprintf('%s_TM_V.bmp', imgFileName(1:end-4));

srcIm = imread(imgFileName);
Im = im2double(srcIm);
[Im_n,Im_m,Im_d]=size(Im);

if(Im_d ~= 3)
    error('Only work on colored images!')
    exit;
end

filterSize=[15,15];
numClusters=25;
filtrerBank=getFilterBank(filterSize);
textonizedIm=uint8(imgTextonization(Im, filtrerBank, numClusters));
textonizedIm=im2double(textonizedIm);
textonizedIm=orderImgTextonization(textonizedIm);
uniqueVals_Textonized=unique(textonizedIm);
imwrite(textonizedIm, outImgFileName);
textonizedIm_View=textonizedIm;
textonizedIm_View=imadjust(textonizedIm_View,[min(textonizedIm_View(:)),max(textonizedIm_View(:))],[0,1]);
imwrite(textonizedIm_View, outImgFileNameForView);
% validity test (save vs. load)
savedTextonizedIm=imread(outImgFileName);
savedTextonizedIm=im2double(savedTextonizedIm);
uniqueVals_Textonized_saved=unique(savedTextonizedIm);
%uniqueVals_Textonized==uniqueVals_Textonized_saved % save vs. load validity check
savedTextonizedIm=imadjust(savedTextonizedIm,[min(savedTextonizedIm(:)),max(savedTextonizedIm(:))],[0,1]);

figure;
subplot(2,2,1); imshow(Im);title('Image');
subplot(2,2,2); imagesc(textonizedIm_View);title('Textonized image');
subplot(2,2,4); imhist(textonizedIm_View);title('Textons dist.');axis([-0.1,1.1,0,1000+(sum(sum(textonizedIm_View==mode(mode(textonizedIm_View)))))]);



