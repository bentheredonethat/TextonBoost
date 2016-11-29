%{
testBoundingBox.m

ML founndations - 2012A
Yaniv Bar

%}
clear;

test=zeros(71,107);
border=1;
rectBorder=0.5;
[im_n,im_m,im_d]=size(test);
numOfRandomRects=5;
for rows=1:5:im_n
    for cols=1:5:im_m
        tempTest = test;
        i=[cols,rows];
        im_dim=[im_n,im_m];
        FIXED_BOUNDING_BOX_FLAG=1;
        [boundingBox_tl, boundingBox_br] = calcBoundingBox(i,im_dim,FIXED_BOUNDING_BOX_FLAG);
        tempTest(boundingBox_tl(2),boundingBox_tl(1):boundingBox_br(1)) = border;
        tempTest(boundingBox_br(2),boundingBox_tl(1):boundingBox_br(1)) = border;
        tempTest(boundingBox_tl(2):boundingBox_br(2),boundingBox_tl(1)) = border;
        tempTest(boundingBox_tl(2):boundingBox_br(2),boundingBox_br(1)) = border;  
        for l=1:numOfRandomRects
            [r_tl,r_br]=calcRectBox(boundingBox_tl, boundingBox_br);
            tempTest(r_tl(2),r_tl(1):r_br(1)) = rectBorder;
            tempTest(r_br(2),r_tl(1):r_br(1)) = rectBorder;
            tempTest(r_tl(2):r_br(2),r_tl(1)) = rectBorder;
            tempTest(r_tl(2):r_br(2),r_br(1)) = rectBorder;       
            
            figure(1);
            imshow(tempTest);
            hold on
            plot(i(1),i(2),'*');
            hold off
        end
    end
end