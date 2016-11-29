%{ 
calcModelFeatures.m

Calculate Model Features

ML founndations - 2012A
Yaniv Bar

%}
clear;
DISPLAY_LEVEL=1;
DEBUG_IMAGE_DISPLAY_FLAG=0;
DEBUG_FLAG=1;

textonized_images_inFolder='ClassificationAttempt/TextonizedImages';
textonized_images_inFolder_pwd=strcat(pwd(), '/', textonized_images_inFolder);
textonized_images_inFolder_fl=ls(textonized_images_inFolder);
textonized_images_inFolder_fl=textonized_images_inFolder_fl(3:end-1,:);
textonized_images_inFolder_nFile=size(textonized_images_inFolder_fl, 1);

groundTruth_images_inFolder='ClassificationAttempt/GroundTruth';
groundTruth_images_inFoldere_pwd=strcat(pwd(), '/', groundTruth_images_inFolder);

images_inFolder='ClassificationAttempt/Images';
images_inFoldere_pwd=strcat(pwd(), '/', images_inFolder);

out_folder='ClassificationAttempt/Model';
out_folder_pwd=strcat(pwd(), '/', out_folder);
out_features_base_mat_file='features.mat';
out_nonFeatures_base_mat_file='nonFeatures.mat';
out_processedClassColors = 'pcc.mat';

colorClasses=initGroundTruthColorClasses();
globalColorFeaturesMappingCell=cell(length(colorClasses),1);


% --- My code -- %%
dir_struct = dir(textonized_images_inFolder);
num_all_files = length(dir_struct)-3;

all_files = cell(1,num_all_files);
temp_files =  cell(1,num_all_files);

count = 1;
for i = 1: length(dir_struct)
    if dir_struct(i).isdir == 0
       
        temp_files{count} = dir_struct(i).name;
        count = count + 1;
    end
end
all_files =  temp_files; 

all_files;
textonized_images_inFolder_nFile = num_all_files;
textonized_images_inFolder_fl = all_files;
% --- end my code--- 

for i=1:textonized_images_inFolder_nFile
    maskSize = [3,3];
    
    textonized_Im_fname= char(textonized_images_inFolder_fl(i));
    textonized_Im_fname=strtrim(textonized_Im_fname);
    groundTruth_Im_fname=strrep(textonized_Im_fname,'_TM','_GT');
    im_fname=strrep(textonized_Im_fname,'_TM','');
    % check if there is a suffix file name problem
    if(length(textonized_Im_fname)<=4 || strcmp(textonized_Im_fname(end-3:end), '.bmp')==false)
        continue;
    end
    
    % process
    if(DISPLAY_LEVEL==1)
            fprintf('\n*--- Processing files %s,%s,%s (%i/%i)',im_fname,textonized_Im_fname,groundTruth_Im_fname,i,textonized_images_inFolder_nFile);
    end
    
    im_pathname=strcat(images_inFoldere_pwd, '/', im_fname);
    im = imread(im_pathname);
    im = im2double(im);
    
    textonized_Im_pathname=strcat(textonized_images_inFolder_pwd, '/', textonized_Im_fname);
    textonized_Im = imread(textonized_Im_pathname);
    textonized_Im = im2double(textonized_Im);
    textonized_Im=imadjust(textonized_Im,[min(textonized_Im(:)),max(textonized_Im(:))],[0,1]);
    Textons=unique(textonized_Im);
    textonized_Im = subSampleImg(textonized_Im,maskSize);
    
    groundTruth_Im_pathname=strcat(groundTruth_images_inFoldere_pwd, '/', groundTruth_Im_fname);
    groundTruth_Im = imread(groundTruth_Im_pathname);
    groundTruth_Im = im2double(groundTruth_Im);
    groundTruth_Im = subSampleImg(groundTruth_Im,maskSize);
    
    if(DEBUG_IMAGE_DISPLAY_FLAG==1)
        figure;
        subplot(1,3,1); imshow(im);
        subplot(1,3,2); imshow(textonized_Im);
        subplot(1,3,3); imshow(groundTruth_Im);
    end
    
    textonizedImgFeaturesCell = calcImgFeatures(Textons,textonized_Im);
    imgColorFeaturesMappingCell=extractImgColorFeaturesMapping(colorClasses, groundTruth_Im, textonizedImgFeaturesCell);
    if(DEBUG_FLAG==1)
        imgColorFeaturesMappingCell;
        for k=1:length(colorClasses)
            imgColorFeatures = imgColorFeaturesMappingCell{k,1};
            [n,len]=size(imgColorFeatures);
        end
    end
    
    for k=1:length(colorClasses)
        imgColorFeatures = imgColorFeaturesMappingCell{k,1};
        globalColorFeatures = globalColorFeaturesMappingCell{k,1};
        newColorFeatures = [globalColorFeatures, imgColorFeatures];
        globalColorFeaturesMappingCell{k,1} = newColorFeatures;
        clear imgColorFeatures;
        clear globalColorFeatures;
    end
    
end

if(DEBUG_FLAG==1)
    globalColorFeaturesMappingCell
end

processedClassColors=[];
for k=1:length(colorClasses)
    features = globalColorFeaturesMappingCell{k,1};
    nonFeatures = [];
    [n,numOfFeatures] = size(features);
    if(numOfFeatures > 0)
        processedClassColors=[processedClassColors, k];
        out_features_mat_file = sprintf('%s_%d.mat', out_features_base_mat_file(1:end-4),k);
        out_nonFeatures_mat_file = sprintf('%s_%d.mat', out_nonFeatures_base_mat_file(1:end-4),k);

        for t=1:length(colorClasses)
            if(t == k)
                continue;
            end
            subsetNonFeatures = globalColorFeaturesMappingCell{t,1};
            nonFeatures = [nonFeatures, subsetNonFeatures];
        end
    
        % saving
        if(DISPLAY_LEVEL==1)
            fprintf('\n*--- Saving files %s,%s',out_features_mat_file,out_nonFeatures_mat_file);
            fprintf('\n*--- Num of features=%i, Num of non-features=%i',length(features),length(nonFeatures));
        end
       
        save(strcat(out_folder_pwd,'/',out_features_mat_file), 'features');
        save(strcat(out_folder_pwd,'/',out_nonFeatures_mat_file), 'nonFeatures');
    end
end

% saving
if(DISPLAY_LEVEL==1)
    fprintf('\n*--- Saving files %s',out_processedClassColors);
end
save(strcat(out_folder_pwd,'/',out_processedClassColors), 'processedClassColors');

fprintf('****** CalcModelFeatures.m done********\n');
 


