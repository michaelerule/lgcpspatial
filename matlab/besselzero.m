% Pnm : the m first zeros of the n-bessel (Jn) functions
% Seyed Ebrahim Chalangar, Sharif University of Technology, Tehran, Iran
function Pnm=besselzero(n,m)
B=@(X)besselj(n,X);
x0=1;
Pnm(1:m)=0;
Pnmn(1:m)=0;
for t=1:m
    while Pnm(t)<0.001
        Pnm(t) = fzero(B,x0);
        while (Pnmn(t)-Pnm(t))<0.001
            x0=x0+0.1;
            Pnmn(t) = fzero(B,x0);
        end
    end
end
end