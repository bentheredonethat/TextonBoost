%{
calcRectBox.m 

ML founndations - 2012A
Yaniv Bar

Calculate rectangle boxes given a bounding box.

Input:
boundingBox_tl - tl point
boundingBox_br - br point
numOfBoxes - number of rect boxes
RANDOM_RECT_SELECTION_FLAG - 0/1 (0 - fixed, 1 - random)

Output:
r_tl - rect tl point
r_br - rect br point
%}
function [RectBoxes_tl, RectBoxes_br] = calcRectBoxes(boundingBox_tl, boundingBox_br,numOfBoxes, RANDOM_RECT_SELECTION_FLAG)
RectBoxes_tl=zeros(numOfBoxes,2);
RectBoxes_br=zeros(numOfBoxes,2);

if(RANDOM_RECT_SELECTION_FLAG==1)
    for l=1:numOfBoxes
        [r_tl,r_br]=calcRectBox(boundingBox_tl, boundingBox_br);      
        RectBoxes_tl(l,:)=r_tl;
        RectBoxes_br(l,:)=r_br;
    end
end

if(RANDOM_RECT_SELECTION_FLAG==0)
    rowBoxes = floor(numOfBoxes/2);
    colBoxes = numOfBoxes-rowBoxes;
    bb_n=boundingBox_br(2)-boundingBox_tl(2)+1;
    bb_m=boundingBox_br(1)-boundingBox_tl(1)+1;
    rowBoxHeight = floor(bb_n/rowBoxes);
    colBoxWidth = floor(bb_m/colBoxes);
    
    ind=1;
    for i=1:rowBoxes
        if(mod(i,2)==1)
            r_tl(1)=boundingBox_tl(1);
            r_tl(2)=boundingBox_tl(2)+(i-1)*rowBoxHeight;
            r_br(1)=boundingBox_tl(1)+floor(bb_m/2);
            r_br(2)=boundingBox_tl(2)+(i*rowBoxHeight-1);
            RectBoxes_tl(ind,:)=r_tl;
            RectBoxes_br(ind,:)=r_br;
            ind=ind+1;
        else
            r_tl(1)=boundingBox_tl(1)+floor(bb_m/2);
            r_tl(2)=boundingBox_tl(2)+(i-1)*rowBoxHeight;
            r_br(1)=boundingBox_br(1);
            r_br(2)=boundingBox_tl(2)+(i*rowBoxHeight-1);
            RectBoxes_tl(ind,:)=r_tl;
            RectBoxes_br(ind,:)=r_br;
            ind=ind+1;  
        end
    end
    for i=1:colBoxes
        if(mod(i,2)==0)
            r_tl(1)=boundingBox_tl(1)+(i-1)*colBoxWidth;
            r_tl(2)=boundingBox_tl(2);
            r_br(1)=boundingBox_tl(1)+(i*colBoxWidth-1);
            r_br(2)=boundingBox_tl(2)+floor(bb_n/2);
            RectBoxes_tl(ind,:)=r_tl;
            RectBoxes_br(ind,:)=r_br;
            ind=ind+1;
        else
            r_tl(1)=boundingBox_tl(1)+(i-1)*colBoxWidth;
            r_tl(2)=boundingBox_tl(2)+floor(bb_n/2);
            r_br(1)=boundingBox_tl(1)+(i*colBoxWidth-1);
            r_br(2)=boundingBox_br(2);
            RectBoxes_tl(ind,:)=r_tl;
            RectBoxes_br(ind,:)=r_br;
            ind=ind+1;     
        end
    end
end

        