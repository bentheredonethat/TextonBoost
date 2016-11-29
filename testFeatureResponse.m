%{
tesFeatureResponse.m

ML founndations - 2012A
Yaniv Bar

Testing shape filter response

%}
clear;

imgFileName = 'textonizationTestImg.bmp';
srcIm = imread(imgFileName);
Im = im2double(srcIm);
[Im_n,Im_m,Im_d]=size(Im);
filterSize=[35,35];
numClusters=5;
filtrerBank=getFilterBank(filterSize);
textonizedIm=uint8(imgTextonization(Im, filtrerBank, numClusters));
textonizedIm=im2double(textonizedIm);
textonizedIm=imadjust(textonizedIm,[min(textonizedIm(:)),max(textonizedIm(:))],[0,1]);
textons=unique(textonizedIm)
numTextons=length(textons);

figure;
subplot(1,2,1); imagesc(textonizedIm);
subplot(1,2,2); imhist(textonizedIm);

% test 1
tempTextonizedIm=textonizedIm;
i=[111,68];
r_tl=[76,27];
r_br=[119,71];
t=textons(5);
border = 0.3;

tempTextonizedIm(r_tl(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_br(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_tl(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_br(1)) = border;  

textonLayerIm=getTextonLayerImg(t,textonizedIm);
textonIntIm=calcIntImg(textonLayerIm);

figure;
subplot(1,2,1); imagesc(tempTextonizedIm);title('Shape filter example on textonized image');
hold on
plot(i(1),i(2),'*');
hold off
tempTitle = sprintf('Layered texton image for texton index=%f (red=texton layer)',t);
subplot(1,2,2); imagesc(textonLayerIm);title(tempTitle);
[featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm)

% test 2
tempTextonizedIm=textonizedIm;
i=[111,68];
r_tl=[76,37];
r_br=[119,91];
t=textons(5);
border = 0.3;

tempTextonizedIm(r_tl(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_br(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_tl(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_br(1)) = border;  

textonLayerIm=getTextonLayerImg(t,textonizedIm);
textonIntIm=calcIntImg(textonLayerIm);

figure;
subplot(1,2,1); imagesc(tempTextonizedIm);title('Shape filter example on textonized image');
hold on
plot(i(1),i(2),'*');
hold off
tempTitle = sprintf('Layered texton image for texton index=%f (red=texton layer)',t);
subplot(1,2,2); imagesc(textonLayerIm);title(tempTitle);

[featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm)

% test 3
tempTextonizedIm=textonizedIm;
i=[111,68];
r_tl=[158,103];
r_br=[195,118];
t=textons(5);
border = 0.7;

tempTextonizedIm(r_tl(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_br(2),r_tl(1):r_br(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_tl(1)) = border;
tempTextonizedIm(r_tl(2):r_br(2),r_br(1)) = border;  

textonLayerIm=getTextonLayerImg(t,textonizedIm);
textonIntIm=calcIntImg(textonLayerIm);

figure;
subplot(1,2,1); imagesc(tempTextonizedIm);title('Shape filter example on textonized image');
hold on
plot(i(1),i(2),'*');
hold off
tempTitle = sprintf('Layered texton image for texton index=%f (red=texton layer)',t);
subplot(1,2,2); imagesc(textonLayerIm);title(tempTitle);

[featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm)