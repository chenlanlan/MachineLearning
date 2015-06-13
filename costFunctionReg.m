function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
origin = X * theta;
preictions = ones(m, 1);
cost = zeros(m, 1);
for i = 1:m,
	predictions(i) = 1/ (1 + 2.71828 ^(- origin(i)));
	cost(i) = -y(i) * log(predictions(i)) - (1 - y(i)) * log(1 - predictions(i));
end
theta2 = theta.^2;
J = 1/m * sum(cost) + lambda /(2 * m) * sum(theta2(2:length(theta2)));
cha = predictions' - y;
add = theta;
add(1) = 0;
g = X' * cha + lambda * add;
grad = 1 / m * g;




% =============================================================

end
