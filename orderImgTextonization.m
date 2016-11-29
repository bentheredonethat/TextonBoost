%{
orderImgTextonization.m 

Order image textonization by apperance rank

ML founndations - 2012A
Yaniv Bar

Input:
textonizedIm

Output:
orderedTextonizedIm - Ordered textonized image
%}
function orderedTextonizedIm = orderImgTextonization(textonizedIm)
orderedTextonizedIm=textonizedIm;
textons=unique(textonizedIm);
[counts,newTextons]=imhist(textonizedIm);
Ind=[];
for i=1:length(textons)
    Ind=[Ind,find(newTextons==textons(i))];
end
counts=counts(Ind);
newTextons=newTextons(Ind);
[sortCounts,Ind]=sort(counts);
newTextons=textons(Ind);
for i=1:length(textons)
    repInd=find(textonizedIm(:)==newTextons(i));
    orderedTextonizedIm(repInd)=textons(i);
end

