function [X] = proxF_tSVD(Y, rho)

[n1, n2, n3] = size(Y);

Y = fft(Y,[],3);
X = zeros(size(Y));

for i = 1:n3
    [U1,S1,V1] = svd(Y(:,:,i));
    s = diag(S1);
    jf = 1 - min( rho./abs(s), 1 );
    s = jf .* s;
    X(:,:,i) = U1*diag(s)*V1';
end

X = ifft(X,[],3);


end