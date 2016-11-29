%{
testFilters.m

ML founndations - 2012A
Yaniv Bar

In the image textonization process an input image is convolved with a filter-bank which includes a 17 - dimensional feature filteres.
The feature filtered image bank is made of 3 Gaussians, 4 Laplacian of Gaussians (LoG) and 4 first order derivatives of Gaussians.
The three Gaussian kernels (with sigma = 1, 2, 4) are applied to each H,S,V channel, thus producing 9 filter responses. 
The four LoGs (with sigma = 1, 2, 4, 8) were applied to the V channel only, thus producing 4 filter responses.
The four derivatives of Gaussians were divided into the two x and y aligned sets, each with two different
values of kernels (sigma = 2, 4). Derivatives of Gaussians were also applied to the V channel only, thus producing 4 final filter
responses. 

This program tests the filters.
%}

clear;

imgFileName = 'textonizationTestImg.bmp';
srcIm = imread(imgFileName);
Im = im2double(srcIm);
[Im_n,Im_m,Im_d]=size(Im);

if(Im_d ~= 3)
    error('Only work on colored images!')
    exit;
else
    Im_hsv = rgb2hsv(Im);
    Im_h = Im_hsv(:,:,1);
    Im_s = Im_hsv(:,:,2);
    Im_v = Im_hsv(:,:,3);
end

resIm = cat(3,Im_h,Im_s,Im_v);
resIm=hsv2rgb(resIm);

figure;
subplot(2,3,1); imagesc(Im);title('HSV decomposition');
subplot(2,3,2); imagesc(resIm);
subplot(2,3,4); imagesc(Im_h);colormap(gray);
subplot(2,3,5); imagesc(Im_s);colormap(gray);
subplot(2,3,6); imagesc(Im_v);colormap(gray);

filterSize=[11,11];
sigma_val1=1;
sigma_val2=2;
sigma_val4=4;
sigma_val8=8;

gaussianFilter = getOneDimGaussianFilter(filterSize,1,'X');
filteredIm_v=conv2(Im_v,gaussianFilter,'same');
figure;
subplot(2,2,1); imagesc(Im_v);colormap(gray);title('1D X-axis gaussian filter example on value component. Sigma value=1.');
subplot(2,2,2); imagesc(filteredIm_v);colormap(gray);
subplot(2,2,3); imagesc(gaussianFilter);colormap(gray);
subplot(2,2,4); mesh(gaussianFilter);

gaussianFilter = getOneDimGaussianFilter(filterSize,1,'Y');
filteredIm_v=conv2(Im_v,gaussianFilter,'same');
figure;
subplot(2,2,1); imagesc(Im_v);colormap(gray);title('1D Y-axis gaussian filter example on value component. Sigma value=1.');
subplot(2,2,2); imagesc(filteredIm_v);colormap(gray);
subplot(2,2,3); imagesc(gaussianFilter);colormap(gray);
subplot(2,2,4); mesh(gaussianFilter);

gaussianFilter = getSymmetricGaussianFilter(filterSize,sigma_val4);
filteredIm_v=conv2(Im_v,gaussianFilter,'same');
figure;
subplot(2,2,1); imagesc(Im_v);colormap(gray);title('1D X&Y-axis gaussian filter example on value component. Sigma value=4.');
subplot(2,2,2); imagesc(filteredIm_v);colormap(gray);
subplot(2,2,3); imagesc(gaussianFilter);colormap(gray);
subplot(2,2,4); mesh(gaussianFilter);

logFilter = getSymmetricLogFilter(filterSize,sigma_val8);
filteredIm_v=conv2(Im_v,logFilter,'same');
figure;
subplot(2,2,1); imagesc(Im_v);colormap(gray);title('1D X&Y-laplacian of gaussian filter example on value component. Sigma value=8.');
subplot(2,2,2); imagesc(filteredIm_v);colormap(gray);
subplot(2,2,3); imagesc(logFilter);colormap(gray);
subplot(2,2,4); mesh(logFilter);colormap(gray);


