function [y] = MLPregressionPredict(w,X,nHidden)
% MLPregressionPredict(w,X,y,nHidden)
%
% Description:
%       - Predicts the Model for the 1D Regression
%
% nHidden:
%       - This is the number of hidden layers in the Neural Network.
%
% Author: Reza Asad (2014)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
    hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
    offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end));

% Compute Output
y = zeros(nInstances,1);
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i) = fp{end}*outputWeights;
end
