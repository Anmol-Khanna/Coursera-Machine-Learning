function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%======= Cost =========================
hyp=sigmoid(X*theta);

%summation=sum(((-y'*log(hyp))-((1-y)'*(log(1-hyp))))); % ...(1)
                                                        %AtB=BtA? y transpose should make it a 1x4 ie horizontal vector so the usual matrix
                                                        %multiplication shouldn't work unless At*B=Bt*A in MATLAB

%summation=sum(((log(hyp)*-y')-((log(1-hyp)*(1-y)')))); % ...(2) Obviously doesn't work since this is A*Bt,
                                                        % My misunderstanding arose from me mentally "naming" log(hyp) as A and y as B...
                                                        % That is not what the law means. The law indicates that either vector can be transposed and the
                                                        % multiplication will still hold good AS LONG AS THE TRANSPOSED VECTOR IS FIRST
                                                        % Try keeping this order but transposing log(hyp)
                                                        % instead of y
                                                 

summation=sum(((log(hyp)'*-y)-((log(1-hyp)'*(1-y))))); % ...(3) Works exactly the same as (1), thus At*B=B*tA proved 
                                                       % Interesting
                                                       % implication that
                                                       % matrix
                                                       % multiplication
                                                       % works even if (or
                                                       % only if) the first
                                                       % vector is
                                                       % horizontal, which
                                                       % of course doesn't
                                                       % hold good with the
                                                       % method we use when
                                                       % computing by hand.
                                                       
                                                       %UNLESS, arrays, by
                                                       %default,are
                                                       %represented in row
                                                       %form (as does seem to be the case- try declaring a new array in the command window). If this is
                                                       %the case, then this
                                                       %wa all just a huge
                                                       %waste of time,
                                                       %because the first
                                                       %array necessarily
                                                       %being transposed
                                                       %would make it a
                                                       %vertical vector
                                                       %always, therefore
                                                       %always being in
                                                       %agreement with the
                                                       %"matrix
                                                       %multiplication by
                                                       %hand" method we
                                                       %were taught.
                                                      
                                                     
                                                      

summation_regularised= sum(theta.^2)-(theta(1).^2);

J=((1/m)*summation)+((lambda/(2*m))*summation_regularised);

%======= Gradient======================
error=hyp-y; 
sum_grad=(error'*X); %1x4 matrix needs to be 4x1 for addition later

temp=theta;
temp(1)=0;
grad = ((1/m).*sum_grad'+(lambda/m).*(temp)); %trasnpose sum_grad to allow addition

% =============================================================

grad = grad(:);

end
