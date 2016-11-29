%{
extractImgColorFeaturesMapping.m 

ML founndations - 2012A
Yaniv Bar

extract image color features mapping (array of cells) 

Input:
groundTruthColorClassesCell - ground truth color classes (array of cells)
groundTruthImage - ground truth image
textonizedImgFeaturesCell - textonized img features (array of cells)

Output:
imgColorFeaturesMappingCell - (array of cells)
%}
function imgColorFeaturesMappingCell=extractImgColorFeaturesMapping(groundTruthColorClassesCell, groundTruthImg, textonizedImgFeaturesCell)
[imgFeaturesCell_n, imgFeaturesCell_m]=size(textonizedImgFeaturesCell);
[groundTruthImg_n, groundTruthImg_m, dc]=size(groundTruthImg);
if(imgFeaturesCell_n ~= groundTruthImg_n || imgFeaturesCell_m~=groundTruthImg_m)
    error('Mismatch images dims...');
end

colorClasses = length(groundTruthColorClassesCell);
imgColorFeaturesMappingCell=cell(colorClasses,1);

for i=1:groundTruthImg_n
    for j=1:groundTruthImg_m
        groundTruthPixelColor=zeros(3,1);
        groundTruthPixelColor(1,1)=groundTruthImg(i,j,1);
        groundTruthPixelColor(2,1)=groundTruthImg(i,j,2);
        groundTruthPixelColor(3,1)=groundTruthImg(i,j,3);
        groundTruthPixelColorClass=-1;
        for k=2:colorClasses
            classColor=groundTruthColorClassesCell{k,2};
            if(sum(groundTruthPixelColor==classColor)==3)
                groundTruthPixelColorClass=k;
            end
        end
        if(groundTruthPixelColorClass ~= -1)
           imgPixelFeatures = textonizedImgFeaturesCell{i,j};
           imgColorFeatures = imgColorFeaturesMappingCell{groundTruthPixelColorClass,1};
           imgColorFeatures = [imgColorFeatures, imgPixelFeatures];
           imgColorFeaturesMappingCell{groundTruthPixelColorClass,1} = imgColorFeatures;
        end
    end
end


