function [Z] = SCLRSmC(Y, M, lambda1, lambda2)

% INPUT:
% Y: sample imaging data, every lateral slice in Y corresponds to one sample
% M: dissimilarity matrix calculated from the data
% lambda1, lambda2: regularization parameters
%
% OUTPUT:
% Z: coefficient tensor
%--------------------------------------------------------------------------
% Copyright @ Tong Wu and Waheed U. Bajwa, 2021
%--------------------------------------------------------------------------

[n1, n2, n3] = size(Y);
Yhat = fft(Y,[],3);

Z = zeros(n2,n2,n3);
C = zeros(n2,n2,n3);
Q = zeros(n2,n2,n3);
oldZ = Z; oldC = C; oldQ = Q;
G1 = zeros(n2,n2,n3);
G2 = zeros(n2,n2,n3);
mu = 0.1;
rho = 1.9;
mu_max = 1e10;
epsilon = 1e-5;
maxIter = 500;
iter = 0;

while iter<maxIter
    iter = iter + 1;
    
    % update C
    A = Z + G1/mu;
    C = proxF_tSVD(A, 1/mu);
    
    % update Q
    B = Z + G2/mu;
    for i = 1:n3
        Q(:,:,i) = max(0,(abs(B(:,:,i)) - lambda1/mu*M) ) .* sign(B(:,:,i));
    end
    
    % update Z
    P1 = C - G1/mu;
    P2 = Q - G2/mu;
    P1hat = fft(P1,[],3);
    P2hat = fft(P2,[],3);
    Zhat = zeros(size(C));
    for i = 1:n3
        Zhat(:,:,i) = (2*lambda2*Yhat(:,:,i)'*Yhat(:,:,i) + 2*mu*eye(n2))\(2*lambda2*Yhat(:,:,i)'*Yhat(:,:,i) + mu*(P1hat(:,:,i) + P2hat(:,:,i)));
    end
    Z = ifft(Zhat,[],3);
    
    stopC = max( [max(max(max(abs(Z - C)))), max(max(max(abs(Z - Q)))), max(max(max(abs(Z - oldZ)))), max(max(max(abs(C - oldC)))), max(max(max(abs(Q - oldQ))))] );
    
    if stopC<epsilon
        break;
    else
        G1 = G1 + mu*(Z - C);
        G2 = G2 + mu*(Z - Q);
        mu = min(rho*mu, mu_max);
        oldZ = Z; oldC = C; oldQ = Q;
    end
    
end

end