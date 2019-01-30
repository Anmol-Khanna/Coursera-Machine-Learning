
function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);   %5000
num_labels = size(all_theta, 1);    %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);           %column vector of size 5000, each element will be eihter a 0 or a 1

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% We have all the weights for each possible output class in all_theta. We
% have a list of digits to be classified, in X. Therefore, attaching each
% value in X to all the weights in all_theta would yield a matrix where the
% largest member denotes the greatest probability of the digit being that
% member's index value (other than '0', which is mapped to index 10 due to 
% the lack of an actual 0th index position).
% So, all_theta is 10x401 and X is 5000x401. Thus, for matrix
% multiplication, intermediate_prediction=all_theta*X'; will yield a
% 10x5000 matrix. 
% p must agree in matrix dimensions with y, because we use the logical
% matrix equivalence right after this function call in ex3.m, ie p must be
% a 5000x1 matrix, which is a list of the labels for all 5000 training examples in X.

intermediate_prediction=all_theta*X';
[M, I]=max(intermediate_prediction);        % Returns max element of each column into a new matrix M,
                                            % and then stores the index
                                            % value of these max elements
                                            % into Vector I, which
                                            % therefore is of the
                                            % dimensions 1x5000

p=I';                                       % I is transposed to match matrix dimensions with y, which is necessary for the 
                                            % logical array equivalency
                                            % carried out in the next line
                                            % in ex3.m


% =========================================================================


end
