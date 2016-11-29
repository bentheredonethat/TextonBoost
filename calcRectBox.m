%{
calcRectBox.m 

ML founndations - 2012A
Yaniv Bar

Calculate rectangle box given a bounding box.

Input:
boundingBox_tl - tl point
boundingBox_br - br point

Output:
r_tl - rect tl point
r_br - rect br point
%}
function [r_tl, r_br] = calcRectBox(boundingBox_tl, boundingBox_br)
min_rect_n=4;
min_rect_m=4;
valid_flag=0;
while(valid_flag==0)
    r_tl=zeros(1,2);
    r_br=zeros(1,2);
    randPoint = rand(1);
    r_tl(1)=boundingBox_tl(1)+ceil(randPoint*(boundingBox_br(1)-boundingBox_tl(1)));
    randPoint = rand(1);
    r_tl(2)=boundingBox_tl(2)+ceil(randPoint*(boundingBox_br(2)-boundingBox_tl(2)));
    r_tl=[r_tl(1),r_tl(2)];
    randPoint = rand(1);
    r_br(1)=r_tl(1)+ceil(randPoint*(boundingBox_br(1)-r_tl(1)));
    randPoint = rand(1);
    r_br(2)=r_tl(2)+ceil(randPoint*(boundingBox_br(2)-r_tl(2)));
    r_br=[r_br(1),r_br(2)];
    rect_n=r_br(2)-r_tl(2)+1;
    rect_m=r_br(1)-r_tl(1)+1;
    if(rect_n>=min_rect_n && rect_m>=min_rect_m)
        valid_flag=1;
    end
end