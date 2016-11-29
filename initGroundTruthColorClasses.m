%{
initGroundTruthColorClasses.m 

ML founndations - 2012A
Yaniv Bar

ground truth images color classes initialization

Input:
none

Output:
groundTruthColorClassesCell - ground truth color classes (array of cells)
%}
function groundTruthColorClassesCell=initGroundTruthColorClasses()
numOfColorsClasses=24;
groundTruthColorClassesCell=cell(numOfColorsClasses,2);
groundTruthColorClassesCell{1,1}='void';
groundTruthColorClassesCell{1,2}=[0;0;0]./255;
groundTruthColorClassesCell{2,1}='building';
groundTruthColorClassesCell{2,2}=[128;0;0]./255;
groundTruthColorClassesCell{3,1}='grass';
groundTruthColorClassesCell{3,2}=[0;128;0]./255;
groundTruthColorClassesCell{4,1}='tree';
groundTruthColorClassesCell{4,2}=[128;128;0]./255;
groundTruthColorClassesCell{5,1}='cow';
groundTruthColorClassesCell{5,2}=[0;0;128]./255;
groundTruthColorClassesCell{6,1}='horse';
groundTruthColorClassesCell{6,2}=[128;0;128]./255;
groundTruthColorClassesCell{7,1}='sheep';
groundTruthColorClassesCell{7,2}=[0;128;128]./255;
groundTruthColorClassesCell{8,1}='sky';
groundTruthColorClassesCell{8,2}=[128;128;128]./255;
groundTruthColorClassesCell{9,1}='mountain';
groundTruthColorClassesCell{9,2}=[64;0;0]./255;
groundTruthColorClassesCell{10,1}='aeroplane';
groundTruthColorClassesCell{10,2}=[192;0;0]./255;
groundTruthColorClassesCell{11,1}='water';
groundTruthColorClassesCell{11,2}=[64;128;0]./255;
groundTruthColorClassesCell{12,1}='face';
groundTruthColorClassesCell{12,2}=[192;128;0]./255;
groundTruthColorClassesCell{13,1}='car';
groundTruthColorClassesCell{13,2}=[64;0;128]./255;
groundTruthColorClassesCell{14,1}='bicycle';
groundTruthColorClassesCell{14,2}=[192;0;128]./255;
groundTruthColorClassesCell{15,1}='flower';
groundTruthColorClassesCell{15,2}=[64;128;128]./255;
groundTruthColorClassesCell{16,1}='sign';
groundTruthColorClassesCell{16,2}=[192;128;128]./255;
groundTruthColorClassesCell{17,1}='bird';
groundTruthColorClassesCell{17,2}=[0;64;0]./255;
groundTruthColorClassesCell{18,1}='book';
groundTruthColorClassesCell{18,2}=[128;64;0]./255;
groundTruthColorClassesCell{19,1}='chair';
groundTruthColorClassesCell{19,2}=[0;192;0]./255;
groundTruthColorClassesCell{20,1}='road';
groundTruthColorClassesCell{20,2}=[128;64;128]./255;
groundTruthColorClassesCell{21,1}='cat';
groundTruthColorClassesCell{21,2}=[0;192;128]./255;
groundTruthColorClassesCell{22,1}='dog';
groundTruthColorClassesCell{22,2}=[128;192;128]./255;
groundTruthColorClassesCell{23,1}='body';
groundTruthColorClassesCell{23,2}=[64;64;0]./255;
groundTruthColorClassesCell{24,1}='boat';
groundTruthColorClassesCell{24,2}=[192;64;0]./255;
