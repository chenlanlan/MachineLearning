function grad = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
origin = X * theta;
preictions = ones(m, 1);
d1 = zeros(m, 1);
d2 = zeros(m, 1);
for i = 1:m,
	predictions(i) = 1/ (1 + 2.71828 ^(- origin(i)));
	d1(i) = (predictions(i) - y(i));
	d2(i) = (predictions(i) - y(i)) * X(i, 2);
grad(1) = 1 / m * sum(d1);
grad(2) = 1 / m * sum(d2);