%{ 
trainModel.m

Train Model

ML founndations - 2012A
Yaniv Bar

%}
clear;
DISPLAY_LEVEL=1;
DEBUG_FLAG=0;

in_folder='ClassificationAttempt/Model';
in_folder_pwd=strcat(pwd(), '/', in_folder);
in_features_base_mat_file='features.mat';
in_nonFeatures_base_mat_file='nonFeatures.mat';
in_processedClassColors_mat_file = 'pcc.mat';
out_folder='ClassificationAttempt/Model';
out_folder_pwd=strcat(pwd(), '/', out_folder);
out_model_base_mat_file='model.mat';

temp = load(strcat(in_folder_pwd,'/',in_processedClassColors_mat_file));
processedClassColors = temp.processedClassColors;
for i=1:length(processedClassColors)
    processedColor=processedClassColors(i);
    in_features_mat_file = sprintf('%s_%d.mat', in_features_base_mat_file(1:end-4),processedColor);
    in_nonFeatures_mat_file = sprintf('%s_%d.mat', in_nonFeatures_base_mat_file(1:end-4),processedColor);
    temp = load(strcat(in_folder_pwd,'/',in_features_mat_file));
    features = temp.features;
    temp = load(strcat(in_folder_pwd,'/',in_nonFeatures_mat_file));
    nonFeatures = temp.nonFeatures;   
    
    if(DISPLAY_LEVEL==1)
        fprintf('\n*--- Train color class phase (%i/%i)',i,length(processedClassColors));
        fprintf('\n*--- Load file %s,%s',in_features_mat_file,in_nonFeatures_mat_file);
        [n,m]=size(features);
        fprintf('\n*--- Features size %iX%i',n,m);  
        [n,m]=size(nonFeatures);
        fprintf('\n*--- Non Features size %iX%i',n,m);  
    end
    
    [n,m]=size(features);
    numFeatures = m;
    [n,m]=size(nonFeatures);
    numNonFeatures = m;
    X=[features, nonFeatures];
    Y=[ones(numFeatures,1);-ones(numNonFeatures,1)];
    if(DISPLAY_LEVEL==1)
        fprintf('\n*--- Train for color class %i',processedColor);  
        [n,m]=size(X);
        fprintf('\n*--- X size %iX%i',n,m);  
        [n,m]=size(Y);
        fprintf('\n*--- Y size %iX%i',n,m);  
        fprintf('\n');
    end
    
    % obtain model
    model = CLSgentleBoost(X,Y);
    out_model_mat_file = sprintf('%s_%d.mat', out_model_base_mat_file(1:end-4),processedColor);
    if(DISPLAY_LEVEL==1)
         fprintf('\n*--- Saving model file %s',out_model_mat_file);
    end
    save(strcat(out_folder_pwd,'/',out_model_mat_file), 'model');   
end

fprintf('\n **** Training Done ******\n');