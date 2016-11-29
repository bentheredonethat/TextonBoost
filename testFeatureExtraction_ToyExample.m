%{
testFeatureExtraction_ToyExample.m

ML founndations - 2012A
Yaniv Bar

Testing shape filter response on toy example

%}
clear;

DEBUG_FLAG=0;

imgFileName = 'toyExample.bmp';
textonizedImgFileName = 'toyExample_TM.bmp';

im=imread(imgFileName);
im=im2double(im);

textonizedIm=imread(textonizedImgFileName);
textonizedIm=im2double(textonizedIm);
[im_n,im_m,im_d]=size(textonizedIm);
if(im_d == 3)
    textonizedIm = rgb2gray(textonizedIm);
end
im_dim=[im_n,im_m];
textons=unique(textonizedIm)
numClusters=length(textons);

figure(1);
subplot(1,3,1); image(im);title('toy examle');
subplot(1,3,2); imagesc(textonizedIm);title('toy examle textonized image');
subplot(1,3,3); imhist(textonizedIm);title('toy examle textonized image distribution');

t=textons(3);
textonLayerIm=getTextonLayerImg(t,textonizedIm);
textonIntIm=calcIntImg(textonLayerIm);
featureExtractionIm = zeros(im_n, im_m);
for i=1:1:im_n
    for j=1:1:im_m
        pixel=[j,i];
        FIXED_BOUNDING_BOX_FLAG=0;
        % fixed bounding box bounda the pixel about half of the image area
        % in both dims.
        [boundingBox_tl, boundingBox_br] = calcBoundingBox(pixel,im_dim,FIXED_BOUNDING_BOX_FLAG);
        FIXED_RECT_FLAG=1;
        if(FIXED_RECT_FLAG==1)
            bb_n=boundingBox_br(2)-boundingBox_tl(2)+1;
            bb_m=boundingBox_br(1)-boundingBox_tl(1)+1;
            % the rectangular mask is the top-left rectangular portion of
            % the pixel
            r_tl(1)=boundingBox_tl(1);
            r_tl(2)=boundingBox_tl(2);
            r_br(1)=boundingBox_tl(1)+floor(bb_m*(1/2));
            r_br(2)=boundingBox_tl(2)+floor(bb_n*(1/2));
        else
            [r_tl,r_br]=calcRectBox(boundingBox_tl, boundingBox_br);
        end
        [featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm);
        featureExtractionIm(i,j)=featureResponsePerc;
        if(DEBUG_FLAG==1)
            figure(2);
            subplot(1,3,1); imshow(textonizedIm);
            hold on
                plot(j,i,'*');
            hold off
            subplot(1,3,2); imshow(textonLayerIm);
            subplot(1,3,3); imshow(featureExtractionIm);
        end
    end
end


figure(3);
subplot(1,3,1); imagesc(textonizedIm);title('toy examle textonized image');
tempTitle = sprintf('Layered texton image for texton index=%f (red=texton layer)',t);
subplot(1,3,2); imagesc(textonLayerIm);title(tempTitle);
subplot(1,3,3); imagesc(featureExtractionIm);title('toy examle feature response image (warm colors=higer response value)');



