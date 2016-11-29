%{
imagesTextonization.m

ML founndations - 2012A
Yaniv Bar

%}
clear;
DISPLAY_LEVEL=1;

images_LimitFlag=31;
images_PrefixFlag=0;
images_fl_Limit= 31;
images_fl_strat_prefix='4_';
images_fl_strat_prefix_len=length(images_fl_strat_prefix);
out_files_delete_flag=0;

if(DISPLAY_LEVEL==1)
    display('* Initialization...')
end

images_inFolder='ClassificationAttempt/Images';
images_inFoldere_pwd=strcat(pwd(), '/', images_inFolder);
images_inFolder_fl= ls(images_inFolder);
images_inFolder_fl=images_inFolder_fl(1:end,:);
images_inFolder_nFile=size(images_inFolder_fl, 1);
images_inFolder_nFile_filt=0;

textonizedImages_outFolder='ClassificationAttempt/TextonizedImages';
textonizedImages_outFolder_pwd=strcat(pwd(), '/', textonizedImages_outFolder);
textonizedImages_outFolder_fl=ls(textonizedImages_outFolder);
textonizedImages_outFolder_fl=textonizedImages_outFolder_fl(3:end,:);
textonizedImages_outFolder_nFile=size(textonizedImages_outFolder_fl, 1);

textonizedImages_View_outFolder='ClassificationAttempt/TextonizedImages/View';
textonizedImages_View_outFolder_pwd=strcat(pwd(), '/', textonizedImages_View_outFolder);

% delete existing textonized images (based on out_files_delete_flag flag)
if(out_files_delete_flag==1)
    if(DISPLAY_LEVEL==1)
        display('* Delete existing textonized images...')
    end
    for i=1:textonizedImages_outFolder_nFile
        out_fname=textonizedImages_outFolder_fl(i,:);
        out_pathname=strcat(textonizedImages_outFolder_pwd, '/', out_fname);  
        delete(out_pathname);
    end
end

for i=1:images_inFolder_nFile
    fname=images_inFolder_fl(i,:);
    fname=strtrim(fname);    
   
    
    % check if there is a prefix file name process limitation
    if(images_PrefixFlag==1)
        prefix_fname=fname(1:images_fl_strat_prefix_len);
        if(strcmp(images_fl_strat_prefix, prefix_fname)==false)
            continue;
        end
    end
    % check if there is a suffix file name problem
    if(length(fname)<=4 || strcmp(fname(end-3:end), '.bmp')==false)
        continue;
    end
   
    images_inFolder_nFile_filt=images_inFolder_nFile_filt+1;
    images_inFolder_fl_filt(images_inFolder_nFile_filt,:)=images_inFolder_fl(i,:);
end
images_inFolder_fl=images_inFolder_fl_filt
images_inFolder_nFile=images_inFolder_nFile_filt
clear images_inFolder_fl_filt;
clear images_inFolder_nFile_filt;
    
% limit maximum possible number of files images (based on images_LimitFlag flag).
if(images_LimitFlag==1)
    if(images_fl_Limit<images_inFolder_nFile)
        images_inFolder_nFile = images_fl_Limit;
        images_inFolder_fl = images_inFolder_fl(1:images_inFolder_nFile,:);
    end
end

if(DISPLAY_LEVEL==1)
    fprintf('* Number of images to process = %i',images_inFolder_nFile);
    images_inFolder_fl
end

% start: Should be the same as in testModel.m
filterSize=[5,5];
numClusters=3;
% end
if(DISPLAY_LEVEL==1)
    display('* Get filter bank...')
end
filtrerBank=getFilterBank(filterSize);

if(DISPLAY_LEVEL==1)
    display('* Performing images textonization...')
end


% --- My code -- %%
dir_struct = dir(images_inFolder);


mysize = length(dir_struct);
remove = zeros(mysize,1, 'logical');

for i=1:mysize % search struct for only file names with '.bmp'  
    % if file name dosen't contain .bmp remove it 
    if length(strfind(dir_struct(i).name,'.bmp')) <= 0  
        remove(i) = 1;
    end
end
index = find(remove == 1);
dir_struct(index) = []; % remove files not '.bmp'
   


num_all_files = length(dir_struct);
all_files = cell(1,num_all_files);

count = 1;
for i=1:num_all_files
    if dir_struct(i).isdir == 0 && strcmp(dir_struct(i).name, '.DS_Store') == 0
        
        all_files{count} = dir_struct(i).name; 
        count = count+1;
    end
end
images_inFolder_nFile = num_all_files;
% --- end my code--- 


for i=1:images_inFolder_nFile
    fname= char(all_files(i))%images_inFolder_fl(i,:);
    fname=strtrim(fname);    
           
    % process
    if(DISPLAY_LEVEL==1)
            fprintf('\n*--- Processing file = %s',fname);
    end
    
    pathname=strcat(images_inFoldere_pwd, '/', fname);
    srcIm = imread(pathname);
    Im = im2double(srcIm);
    [Im_n,Im_m,Im_d]=size(Im);
    if(Im_d ~= 3)
        error('Only work on colored images!')
        exit;
    end
    out_fname= strrep(fname,'.bmp','_TM.bmp');
    if(DISPLAY_LEVEL==1)
            fprintf('\n*--- Writing file = %s',out_fname);
    end
    textonizedIm=uint8(imgTextonization(Im, filtrerBank, numClusters));
    textonizedIm=im2double(textonizedIm);
    textonizedIm=orderImgTextonization(textonizedIm);
    out_pathname=strcat(textonizedImages_outFolder_pwd, '/', out_fname);  
    imwrite(textonizedIm, out_pathname);
    
    out_fname_View = strrep(fname,'.bmp','_TM_V.bmp');
    textonizedIm_View=textonizedIm;
    textonizedIm_View=imadjust(textonizedIm_View,[min(textonizedIm_View(:)),max(textonizedIm_View(:))],[0,1]);
    out_pathname_View=strcat(textonizedImages_View_outFolder_pwd, '/', out_fname_View);  
    imwrite(textonizedIm_View, out_pathname_View);   
    
    if(DISPLAY_LEVEL==1)
        fprintf('\n*--- So far processed = %3.2f perc.', i/images_inFolder_nFile*100);
    end
end
fprintf('\n');

textonized_images_inFolder='ClassificationAttempt/TextonizedImages';
textonized_images_inFolder_pwd=strcat(pwd(), '/', textonized_images_inFolder);
textonized_images_inFolder_fl=ls(textonized_images_inFolder);
textonized_images_inFolder_fl=textonized_images_inFolder_fl(3:end-1,:);
textonized_images_inFolder_nFile=size(textonized_images_inFolder_fl, 1);


% --- My code -- %%
dir_struct = dir(textonized_images_inFolder);
%num_all_files = length(dir_struct)-3;

mysize = length(dir_struct);
remove = zeros(mysize,1, 'logical');

for i=1:mysize % search struct for only file names with '.bmp'  
    % if file name dosen't contain .bmp remove it 
    if length(strfind(dir_struct(i).name,'.bmp')) <= 0  
        remove(i) = 1;
    end
end
index = find(remove == 1);
dir_struct(index) = []; % remove files not '.bmp'
num_all_files = length(dir_struct);

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




handle = figure;

for i=1:textonized_images_inFolder_nFile
    textonized_Im_fname= char(textonized_images_inFolder_fl(i));
    textonized_Im_fname=strtrim(textonized_Im_fname);
    textonized_Im_pathname=strcat(textonized_images_inFolder_pwd, '/', textonized_Im_fname);
    textonized_Im = imread(textonized_Im_pathname);
    [n,m]=size(textonized_Im);
    textonized_Im = im2double(textonized_Im);
    textonized_Im=imadjust(textonized_Im,[min(textonized_Im(:)),max(textonized_Im(:))],[0,1]);
% %     subplot(textonized_images_inFolder_nFile,2,(i-1)*2+1);imshow(textonized_Im);title('Textonized Image');
% %     subplot(textonized_images_inFolder_nFile,2,(i-1)*2+2);imhist(textonized_Im);title('Textons dist.');axis([-0.1,1.1,0,1000+(sum(sum(textonized_Im==mode(mode(textonized_Im)))))]);
end

out_fname_View_fig = 'distFig.fig';
out_pathname_View_fig=strcat(textonizedImages_View_outFolder_pwd, '/', out_fname_View_fig);  
saveas(handle, out_pathname_View_fig);

fprintf('**** imagesTextonization done ****')
clear;


