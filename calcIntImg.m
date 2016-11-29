%{
calcIntImg.m

ML founndations - 2012A
Yaniv Bar

calculate the integral image

Input:
f - image

Output:
g - integral image
%}
function g = calcIntImg(f)
[f_n,f_m]=size(f);
g=zeros(f_n,f_m);
for i=1:f_n
    for j=1:f_m
        tempSum=sum(sum(f(1:i,1:j)));
        g(i,j)=tempSum;
    end
end
