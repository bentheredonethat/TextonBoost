%{
imgTextonization.m

ML founndations - 2012A
Yaniv Bar

Generate image texton map.
This is done by obtaining a 17 - dimensional feature filtered image bank which is clustered using k-means and assigned values.

The feature filtered image bank is made of 3 Gaussians, 4 Laplacian of Gaussians (LoG) and 4 first order derivatives of Gaussians.
The three Gaussian kernels (with sigma = 1, 2, 4) are applied to each H,S,V channel, thus producing 9 filter responses. 
The four LoGs (with sigma = 1, 2, 4, 8) were applied to the V channel only, thus producing 4 filter responses.
The four derivatives of Gaussians were divided into the two x and y aligned sets, each with two different
values of kernels (sigma = 2, 4). Derivatives of Gaussians were also applied to the V channel only, thus producing 4 final filter
responses. 

Therefore, each pixel in each image has associated a 17-dimensional feature vector. These 17-dimensional feature
vectors are then clustered using k-means and assigned values which represents the textonized image.

Input:
Im - a colored image (matrix)
filterBank - 11 dimensional filter bank
numClusters - number or clusters

Output:
textonizedIm - textonized image
%}
function [textonizedIm]=imgTextonization(Im, filterBank,numClusters)
% The training image is convolved with a 17-dimensional filter-bank at scale k
numFeatures = 17;

Im = im2double(Im);
[Im_n,Im_m,Im_d]=size(Im);
Im_hsv = rgb2hsv(Im);
Im_h = Im_hsv(:,:,1);
Im_s = Im_hsv(:,:,2);
Im_v = Im_hsv(:,:,3);

cell{1,1}=conv2(Im_h,filterBank{1,1},'same');
cell{1,2}=conv2(Im_s,filterBank{1,1},'same');
cell{1,3}=conv2(Im_v,filterBank{1,1},'same');
cell{1,4}=conv2(Im_h,filterBank{1,2},'same');
cell{1,5}=conv2(Im_s,filterBank{1,2},'same');
cell{1,6}=conv2(Im_v,filterBank{1,2},'same');
cell{1,7}=conv2(Im_h,filterBank{1,3},'same');
cell{1,8}=conv2(Im_s,filterBank{1,3},'same');
cell{1,9}=conv2(Im_v,filterBank{1,3},'same');
cell{1,10}=conv2(Im_v,filterBank{1,4},'same');
cell{1,11}=conv2(Im_v,filterBank{1,5},'same');
cell{1,12}=conv2(Im_v,filterBank{1,6},'same');
cell{1,13}=conv2(Im_v,filterBank{1,7},'same');
cell{1,14}=conv2(Im_v,filterBank{1,8},'same');
cell{1,15}=conv2(Im_v,filterBank{1,9},'same');
cell{1,16}=conv2(Im_v,filterBank{1,10},'same');
cell{1,17}=conv2(Im_v,filterBank{1,11},'same');

%The 17D responses for all training pixels are then whitened (to give zero mean and unit covariance),
X=zeros(Im_n*Im_m, numFeatures);
for i=1:numFeatures
    pixelFeatureVec = cell{1,i}(:);
    pixelFeatureVec = pixelFeatureVec - mean(pixelFeatureVec);
    pixelFeatureVec=pixelFeatureVec/std(pixelFeatureVec(:));       
    X(:,i) = pixelFeatureVec;
end

[X,rY,initC,finalC]=my_kMeansClustering(X,numClusters,20);
rY=rY(:,numFeatures+1);

%[rY,ctrs] = kmeans(X,numClusters,'Distance','sqEuclidean','Start','sample','Replicates',1,'MaxIter',20); %matlab imp. (alt.)

textonizedIm = reshape(rY, Im_n, Im_m);

