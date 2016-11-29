%{
testCalcImgFeatures.m

ML founndations - 2012A
Yaniv Bar

%}
clear;
DEBUG_FLAG=1;

textonizedImgFileName = 'textonizationTestImg_TM.bmp';
Im = imread(textonizedImgFileName);
Im = im2double(Im);
Im=imadjust(Im,[min(Im(:)),max(Im(:))],[0,1]);
textons=unique(Im)
numTextons=length(textons);
maskSize = [3,3];
Im = subSampleImg(Im,maskSize);
textonizedImgFeaturesCell = calcImgFeatures(textons,Im);

if(DEBUG_FLAG==1)
    numOfTests=3;
    for i=1:numOfTests 
        figure(i);
        subplot(1,1,1); imagesc(Im);
        hold on;
        [x,y]=ginput(1);
        x=ceil(x);
        y=ceil(y);
        hold off;
        figure(i)
        subplot(1,2,1); axis([1,numTextons,0,1]); bar(textonizedImgFeaturesCell{y,x});
        subplot(1,2,2); imagesc(Im);
        hold on
        plot(x,y,'*');
        hold off     
        
        fprintf('\n');
    end
end


