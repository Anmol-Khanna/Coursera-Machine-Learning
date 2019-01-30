function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
summation=0;
sum_reg=0;
hyp=X*theta;  %118*1, just like y

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
   summation=summation+((-y(i).*log(sigmoid(hyp(i)))-(1-y(i)).*(log(1-sigmoid(hyp(i)))))); 
end



error=sigmoid(hyp)-y; %Still 118x1
sum_grad=(error'*X);%1x118 * 118*28 = 1x28

for i=2:n
    sum_reg=sum_reg+theta(i).^2;
end 
J=((1/m)*summation)+((lambda/(2*m))*sum_reg);

for i=1:n
    if i==1
       grad(i)=((1/m).*sum_grad(i)); 
    else
       grad(i)=((1/m).*sum_grad(i)+(lambda/m).*(theta(i)));
    end
end


% =============================================================

end
