function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_1=theta(1);
theta_2=theta(2);
theta_transpose=transpose(theta); %theta_transpose is 1x2
X_transpose= transpose(X); %X transpose is 2x97
hypothesis=theta_transpose*X_transpose; % 1x97
hypothesis_transpose=transpose(hypothesis); %97x1

% for i=1:m
%         intermediate_zero=intermediate_zero+((hypothesis(i)-y(i)).*X(i,1));
%         intermediate_one=intermediate_one+((hypothesis(i)-y(i)).*X(i,2));      
%           sum=sum+((hypothesis_transpose(i)-y(i))*X(i,:)); %1x97-1x97=1x97*97x2=>sum is 1x2, 1st col is sum of x0, 2nd is sum of x1

% a0=(1/m)*(sum(X*theta-y)); 
% a1=(1/m)*(sum((X*theta-y)'*X(:,2))); %1x97 * 97x1
% end
% a=[a0;a1];
% fprintf('intermediate_zero is');
% disp(intermediate_zero);
% fprintf('intermediate one is');
% disp(intermediate_one);
% summation=[intermediate_zero;intermediate_one];
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

% b=alpha*a;
term=alpha/m;
theta_1=theta_1-(term*(sum(X*theta-y))); %theta is 2x1, alpha/m is scalar, sum is 1x2.
theta_2=theta_2-(term*(sum((X*theta-y)'*X(:,2))));
theta=[theta_1;theta_2];
fprintf('Theta is-');
disp(theta);
    % ============================================================
% theta_full=[theta_zero;theta_one];
% disp(theta_full);
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
