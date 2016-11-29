%{
calcImgFeatures.m 

ML founndations - 2012A
Yaniv Bar

calculate textonized image features

Input:
Textons - textons
Im - textonized image

Output:
textonizedImgFeaturesCell - textonized image features (array of cells)
%}
function textonizedImgFeaturesCell=calcImgFeatures(Textons,Im)
DEBUG_FLAG=0;

x=-1;
y=-1;
if(DEBUG_FLAG==1)
    figure(1);
    subplot(1,1,1); imshow(Im);
    hold on;
    [x,y]=ginput(1);
    x=ceil(x);
    y=ceil(y);
    hold off;
end

numOfTextons=length(Textons);
[Im_n,Im_m,Im_d]=size(Im);
textonizedImgFeaturesCell = cell(Im_n,Im_m);

textonizedIntImgCell = cell(length(Textons),1);
for i=1:numOfTextons
    t=Textons(i);
    textonLayerIm=getTextonLayerImg(t,Im);
    textonIntIm=calcIntImg(textonLayerIm);
    textonizedIntImgCell{i,1}=textonIntIm;
end

im_dim=[Im_n,Im_m];
numOfRandomRects=5;
for i=1:Im_n
    for j=1:Im_m
        imgFeatures=zeros(numOfTextons*numOfRandomRects,1);        
        imgFeaturesInd=1;
        pixel=[j,i];
        FIXED_BOUNDING_BOX_FLAG=1;
        [boundingBox_tl, boundingBox_br] = calcBoundingBox(pixel,im_dim,FIXED_BOUNDING_BOX_FLAG);
        RANDOM_RECT_SELECTION_FLAG = 1;
        [RectBoxes_tl,RectBoxes_br]=calcRectBoxes(boundingBox_tl, boundingBox_br,numOfRandomRects, RANDOM_RECT_SELECTION_FLAG);
        for l=1:numOfRandomRects
            r_tl=RectBoxes_tl(l,:);
            r_br=RectBoxes_br(l,:); 
            for k=1:numOfTextons
                textonIntIm=textonizedIntImgCell{k,1};              
                [featureResponse, featureResponsePerc]=calcFeatureResponse(r_tl,r_br,textonIntIm);
                imgFeatures(imgFeaturesInd,1)=featureResponsePerc;
                imgFeaturesInd=imgFeaturesInd+1;
            end    

            if(i==y && j==x && DEBUG_FLAG==1)
                tempTest=Im;
                rectBorder=1;
                tempTest(boundingBox_tl(2),boundingBox_tl(1):boundingBox_br(1)) = rectBorder;
                tempTest(boundingBox_br(2),boundingBox_tl(1):boundingBox_br(1)) = rectBorder;
                tempTest(boundingBox_tl(2):boundingBox_br(2),boundingBox_tl(1)) = rectBorder;
                tempTest(boundingBox_tl(2):boundingBox_br(2),boundingBox_br(1)) = rectBorder;  
                tempTest(r_tl(2),r_tl(1):r_br(1)) = rectBorder;
                tempTest(r_br(2),r_tl(1):r_br(1)) = rectBorder;
                tempTest(r_tl(2):r_br(2),r_tl(1)) = rectBorder;
                tempTest(r_tl(2):r_br(2),r_br(1)) = rectBorder;                       
                numOfEntitiesPerTextonsPass = numOfTextons;
                subSet=imgFeatures(imgFeaturesInd-numOfEntitiesPerTextonsPass:imgFeaturesInd-1,1);      
                figure(l)
                subplot(1,2,1); AXIS([1,numOfTextons,0,1]); bar(subSet);
                subplot(1,2,2); imshow(tempTest);
                hold on
                plot(x,y,'*');
                hold off
                fprintf('\n');
            end                
        end
        textonizedImgFeaturesCell{i,j}=imgFeatures;
    end
end