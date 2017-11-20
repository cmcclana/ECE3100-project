%% SETUP
close all
load demolog1
m = size(X,1);
%% OPTIMIZATION ROUTINES CLASSIFIER WITH TWO FEATURES
% data and params
degree = 10;
X = expand(X(:,1),X(:,2),degree);
bottom = min(X(:,[2,3]));
top = max(X(:,[2,3]));
avg = mean(X(:,2:end));
var = std(X(:,2:end));
% X(:,2:end) = (X(:,2:end) - avg)./var;
T = 1e-4*rand(size(X,2),1);
% fminunc
options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'MaxIterations',400);
lambda = 100;
[T,J] = fminunc(@(T)(cost(T,X,y,lambda)),T,options);
% mesh building
P = 50;
mx = linspace(bottom(1),top(1),P);
my = linspace(bottom(2),top(2),P);
[mx,my] = meshgrid(mx,my);
mesh = expand(mx(:),my(:),degree);
% mesh(:,2:end) = (mesh(:,2:end) - avg)./var;
% plotting results
figure;
v = mesh*T;
gscatter(X(:,2),X(:,3),y);    
hold on;    
contour(reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');
legend('0','1','Boundary');
hold off
grid on
% display params
T'
J
output = sigmoid(X*T);
output(output>=0.5) = 1;
output(output<0.5) = 0;
missed = sum(abs(output-y))
%% FUNCTION TO COMPUTE COST AND GRADIENT
function [J, grad] = cost(T,X,y,lambda)
    m = size(X,1);
    h = sigmoid(X*T);
    J = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
end
%% FUNCTION TO ADD POLYNOMIAL TERMS
function [X] = expand(x1,x2,degree)
    m = size(x1,1);
    X = ones(m,(degree+1)*(degree+2)/2); % (d+1)*(d+2)/2 -> consecutive numbers summation
    n = 2;
    for i = 1:degree
        for j = 0:i
            X(:,n) = (x1.^(i-j)).*(x2.^j);
            n = n + 1; 
        end
    end
end
%% SIGMOID FUNCTION
function f = sigmoid (x)
    f = 1./(1+exp(-x));
end