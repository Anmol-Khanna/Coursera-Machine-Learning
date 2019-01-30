function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
summation=0;
grad_sum_zero=0;
grad_sum_one=0;
grad_sum_two=0;
grad_zero=0;
grad_one=0;
grad_two=0;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
hyp=X*theta;
for i=1:m;    
    summation=summation+(-y(i).*log(sigmoid(hyp(i)))-(1-y(i)).*log(1-sigmoid(hyp(i))));    
end    

J=1/m.*summation;

for i=1:m
    grad_sum_zero=grad_sum_zero+(sigmoid(hyp(i))-y(i))*X(i,1);
    grad_sum_one=grad_sum_one+(sigmoid(hyp(i))-y(i))*X(i,2);
    grad_sum_two=grad_sum_two+(sigmoid(hyp(i))-y(i))*X(i,3);
end
grad_zero=1/m.*(grad_sum_zero);
grad_one=1/m.*(grad_sum_one);
grad_two=1/m.*(grad_sum_two);

grad=[grad_zero;grad_one;grad_two];
% =============================================================

end
