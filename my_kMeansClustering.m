%{
my_kMeansClustering.m

ML founndations - 2012A
Yaniv Bar

kMeans algorithm implementation.

Input: 
X - matrix data: objects in columns and attributes in rows
k - number of clusters
maxIter - max iteration
isRand - random initialization (if true), sequential initialization otherwise

Output:
X - matrix data
rY - a column verctor which represents the group of each object
initC - initial centroids
finalC - final centroids 
%}
function [X,rY,initC,finalC]=my_kMeansClustering(X,k,maxIter,isRand)
if (nargin<4)
    isRand=0;   
else
    isRand=1;
end

[numOfObjects,numOfAtts]=size(X);
% initial value of centroid    
if (isRand==1)
    % random initialization
    p = randperm(k);      
    for i=1:k
        initC(i,:)=X(p(i),:);
    end
else
    % sequential initialization   
    for i=1:k
        initC(i,:)=X(i,:);   
    end
end
finalC=initC;
% initialize as zero vector  
temp=zeros(numOfObjects,1);     
for tempIter=1:maxIter
    % calculate objcets-centroid distances
    d=computeEuclideanDistMatrix(X,finalC);  
    % find group matrix g
    [z,g]=min(d,[],2);  
    if (g==temp)
        % stop the iteration,
        break;  
    else
        % copy group matrix to temporary variable
        temp=g; 
    end
    for i=1:k
        f=find(g==i);
        % only compute centroid if f is not empty
        if f    
            finalC(i,:)=mean(X(find(g==i),:),1);
        end
    end
end    
rY=[X,g];    
end