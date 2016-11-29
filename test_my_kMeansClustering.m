%{ 
test_my_kMeansClustering.m

ML founndations - 2012A
Yaniv Bar
%}

% test (manual clustering example)UG
DEBUG_FLAG=0;
axisVec=[0,6,0,6];
%X=[1,2,4,5,2;1,1,3,4,3]';
X=[1,1;2,1;4,3;5,4]
[numOfObjects,numOfAtts]=size(X);
k=2;
if(DEBUG_FLAG==1)
    figure;
    plot(X(:,1),X(:,2),'go');
    axis(axisVec);
    grid on;
end
% my implementation
[X,rY,initC,finalC]=my_kMeansClustering(X,k,100,0)
% matlab implmentation
[rY2,ctrs] = kmeans(X,k,'Distance','sqEuclidean', 'Start','sample','Replicates',1,'MaxIter',120)

if(DEBUG_FLAG==1 && k==1)
    figure;
    plot(X(:,1),X(:,2),'go',initC(:,1),initC(:,2),'pk');
    axis(axisVec);
    grid on;
    figure;
    plot(X(find(rY(:,numOfAtts+1)==1),1),X(find(rY(:,numOfAtts+1)==1),2),'bs',finalC(:,1),finalC(:,2),'pk');
    axis(axisVec);
    grid on;
end
if(DEBUG_FLAG==1 && k==2)
    figure;
    plot(X(:,1),X(:,2),'go',initC(:,1),initC(:,2),'pk');
    axis(axisVec);
    grid on;
    figure;
    plot(X(find(rY(:,numOfAtts+1)==1),1),X(find(rY(:,numOfAtts+1)==1),2),'bs',X(find(rY(:,numOfAtts+1)==2),1),X(find(rY(:,numOfAtts+1)==2),2),'gs',finalC(:,1),finalC(:,2),'pk');
    axis(axisVec);
    grid on;
end
if(DEBUG_FLAG==1 && k==3)
    figure;
    plot(X(:,1),X(:,2),'go',initC(:,1),initC(:,2),'pk');
    axis(axisVec);
    grid on;
    figure;
    plot(X(find(rY(:,numOfAtts+1)==1),1),X(find(rY(:,numOfAtts+1)==1),2),'bs',X(find(rY(:,numOfAtts+1)==2),1),X(find(rY(:,numOfAtts+1)==2),2),'gs',X(find(rY(:,numOfAtts+1)==3),1),X(find(rY(:,numOfAtts+1)==3),2),'rs',finalC(:,1),finalC(:,2),'pk');
    axis(axisVec);
    grid on;
end
    

