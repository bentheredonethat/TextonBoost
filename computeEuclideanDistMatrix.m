%{
computeEuclideanDistMatrix.m

ML founndations - 2012A
Yaniv Bar

compute euclidean distance matrix

Input: 
A - matrix A
B - matrix B

Output:
d - distance matrix
%}
function d=computeEuclideanDistMatrix(A,B)
[hA,wA]=size(A);
[hB,wB]=size(B);
if (hA==1&& hB==1) 
   d=sqrt(dot((A-B),(A-B)));
else
   C=zeros(wB,hB);
   C(1,:)=ones(1,hB);
   
   %C=[ones(1,hB);zeros(1,hB)];
   D=flipud(C);
   
   E=zeros(wA,hA);
   E(1,:)=ones(1,hA);
   
   %E=[ones(1,hA);zeros(1,hA)];
   
   
   F=flipud(E);
   G=A*C;
   H=A*D;
   I=B*E;
   J=B*F;
   d=sqrt((G-I').^2+(H-J').^2);
end