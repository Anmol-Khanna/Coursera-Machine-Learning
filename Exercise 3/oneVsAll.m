function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);   %size(X,1) returns the number of rows of any matrix X, ie m = 5000, here
n = size(X, 2);   %size(X,2) returns the number of columns of any matrix X, ie n = 400, here

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);       %all_theta is therefore 10x401

% Add ones to the X data matrix
X = [ones(m, 1) X];     %For bias?

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               (ie 10)
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%   We are to pass X,y,lambda (which we have) and each particular theta (after every step of minimising) to lrCostFunction, which will
%   yield the cost J and gradient of that cost. Then, we need to minimise the values we get back from lrCostFunction to find the best possible
%   training accuracy, which we will do through the use of fmincg. The
%   output of fmincg will return the optimal theta, and we loop through for
%   each value in num_labels, ie for each  of the 10 output classes.
%   Thus, we have to first calculate theta and then pass all four values of each of the training
%   examples to 

%   fmincg edits (minimises) the theta matrix, lrCostFunction calculates the cost and
%   gradiaent of that particular theta matrix.

         
for c=1:num_labels               % For each of the 10 output classes
                                % Train a classifier for one class and return the (n+1) learned parameters  
                                  % Training is done through fmincg 
    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost 
    % Set options for fminunc
    % Set Initial theta
     initial_theta = zeros(n + 1, 1);
     options = optimset('GradObj', 'on', 'MaxIter', 50);
     [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options); 
     all_theta(c,:)= theta(:);
end







% =========================================================================


end
