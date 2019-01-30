function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
MSE=0;

theta_transpose=transpose(theta);
X_transpose= transpose(X);
hypothesis=theta_transpose*X_transpose;
% You need to return the following variables correctly
J=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i=1:m
    MSE=MSE+((hypothesis(i)-y(i)).^2);
end
J = MSE/(2*m);



% =========================================================================

end
