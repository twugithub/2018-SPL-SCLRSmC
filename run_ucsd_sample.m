%--------------------------------------------------------------------------
% This is the main function to run the SCLRSmC algorithm for the image
% clustering problem on the UCSD sample dataset with 4 classes.
%
% Version
% --------
% Companion Code Version: 1.0
%
%
% Citation
% ---------
% Any part of this code used in your work should be cited as follows:
%
% T. Wu and W. U. Bajwa, "A low tensor-rank representation approach for clustering of imaging data,"
% IEEE Signal Processing Letters, vol. 25, no. 8, pp. 1196-1200, 2018, Companion Code, ver. 1.0.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% MIT License
%
% Copyright <2022> <Tong Wu and Waheed U. Bajwa>
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
% associated documentation files (the "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
% copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
% OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
% IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------



clearvars;
close all;
clc


load('sampledata.mat');

vecdata = zeros(size(data,1)*size(data,3), size(data,2));
for i = 1:size(data,2)
    temp = data(:,i,:);
    vecdata(:,i) = temp(:);
end

vecdata = vecdata./repmat(sqrt(sum(vecdata.^2)),[size(data,1)*size(data,3) 1]);
L = vecdata'*vecdata;
sigma = mean(mean(1 - L));
M = ones(size(L)) - exp(-(ones(size(L))-L)/sigma);

lambda1 = 0.01;
lambda2 = 0.05;
Z = SCLRSmC(data, M, lambda1, lambda2);
affi = sqrt(sum(Z.^2,3));
affi = affi + affi';
grps = SpectralClustering(affi,clustern);
grps = bestMap(gt,grps);
acc = sum(gt == grps) / length(gt);