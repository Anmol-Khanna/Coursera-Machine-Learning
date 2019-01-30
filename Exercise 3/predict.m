function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);                     % 5000
num_labels = size(Theta2, 1);       % 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);           % column vector of 5000 elements

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 is 25x401 and Theta2 is 10x26.
% To find the network predictions, need to matrix multiply the input values
% with THeta1, sum up and apply an activation function. Then matrix
% multiply this intermeidate 'Layer 2' matrix with Theta2, sum up at each
% output node and apply anohter activation function to get the matrix of
% probabilities. Then, need to run a max function to attain the indices of
% the max elements for each example.

%   Sigmoid function- 
%   function g = sigmoid(z)
%   g = zeros(size(z));
%   denominator=1+exp(-(z));
%   g=1./denominator;

%   Multiplying the X and Theta1 matrices gives an intermediate step where
%   the overall "contribution" of x1 to the next layer (ie, the value of x1
%   multiplied by the weights attached to it) is added to the contributions of
%   x2, then the contibutions of x3, so on. Thus, the matrix multiplication
%   gives the sum of the contributions of all input nodes. Need to then
%   apply an activation function to the intermediate matrix to fulfill the
%   definition of Layer 2, ie a sigmoid applied to the sum of all
%   contributing values and weights to each node in the second layer. It's
%   not so much the values that go to each node that matters, because when
%   training, the weights are literally randomised at the beginning- thus,
%   training is the process of discovering what the weights should be,
%   whereas the nodes they are attached to in the next layer are nothing
%   more than balnk storage spaces to store this information, essentially.
%   So the weights are far more important than the nodes themselves,
%   because all the crucial information pertaining to classification and
%   learned features are embodied by them, wheeras the nodes in the network
%   exist just to provide their services as "summation and activation"
%   hubs. I make note of this because I was intuitively making the
%   assumption that the way the weights mapped to the Hidden Nodes mattered
%   at all- so it was useful to note that it doesn't, since, as mentioned,
%   all the useful info is in the weight matrix itself, and the nodes are
%   all virtually indistinguishable from each other in the absence of the
%   weights- ie, there is nothign inherently "special" about one hidden
%   node compared to any other that would necessitate that weights be connected to them in any
%   specific way. They are just to hold whatever data the weights provide
%   them.


X = [ones(m, 1) X];     % Add column of ones to X for bias, making X 5000x401 now.

[Z1]=X*Theta1';
[denominator1]=1+exp(-(Z1));
[A2]=1./denominator1;

A2 = [ones(m, 1) A2];   % Add column of 1's to A2 for Bias

[Z2]=A2*Theta2';
denominator2=1+exp(-(Z2));
[H]=1./denominator2;

[M,I]=max(H, [], 2);
p=I;


% =========================================================================


end
