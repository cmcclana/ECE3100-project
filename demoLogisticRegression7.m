%% SETUP
close all
load fisheriris
X = meas(:,[1 2]);
m = size(X,1);
degree = 6;
X = expand(X(:,1),X(:,2),degree);
bottom = min(X(:,[2,3]));
top = max(X(:,[2,3]));
avg = mean(X(:,2:end));
var = std(X(:,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;
lambda = 0.01;
T = 1e-5 * rand(size(X,2),3);
Y = false(m,3);
options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'MaxIterations',1000);
P = 50;
mx = linspace(bottom(1),top(1),P);
my = linspace(bottom(2),top(2),P);
[mx,my] = meshgrid(mx,my);
mesh = expand(mx(:),my(:),degree);
mesh(:,2:end) = (mesh(:,2:end) - avg)./var;
%% TRAINING MULTIPLE CLASSIFIERS USING EVERY CLASS AS THE POSSITIVE CLASS AND THE REST AS THE NEGATIVE CLASS
figure;
% setosa
Y(:,1) = strcmp(species,'setosa');
[T(:,1),~] = fminunc(@(T)(cost(T,X,Y(:,1),lambda)),T(:,1),options);
% versicolor
Y(:,2) = strcmp(species,'versicolor');
[T(:,2),~] = fminunc(@(T)(cost(T,X,Y(:,2),lambda)),T(:,2),options);
% virginica
Y(:,3) = strcmp(species,'virginica');
[T(:,3),~] = fminunc(@(T)(cost(T,X,Y(:,3),lambda)),T(:,3),options);
%% PLOTTING BOUNDARIES
cAx = subplot(1,3,1);    
v = mesh*T(:,1);
gscatter(X(:,2),X(:,3),Y(:,1));
hold(cAx,'on');
contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');    
grid(cAx,'on');    
title(cAx,'Positive Class (y=1): Setosa');
hold(cAx,'off');
legend(cAx,'0','1','Boundary');        
cAx = subplot(1,3,2);    
v = mesh*T(:,2);
gscatter(X(:,2),X(:,3),Y(:,2));
hold(cAx,'on');
contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');
grid(cAx,'on');
title(cAx,'Positive Class (y=1): Versicolor');
hold(cAx,'off');
legend(cAx,'0','1','Boundary');    
cAx = subplot(1,3,3);
v = mesh*T(:,3);
gscatter(X(:,2),X(:,3),Y(:,3));
hold(cAx,'on');
contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');
grid(cAx,'on');
title(cAx,'Positive Class (y=1): Virginica');
hold(cAx,'off');
legend(cAx,'0','1','Boundary');
%% PREDICTING THE CLASS FOR A NEW INPUT
while (true)
    % reading input
    [x1,x2] = ginput(1);
    x1_ = (x1 * var(1)) + avg(1);
    x2_ = (x2 * var(2)) + avg(2);
    if (isempty(x1)), break; end    
    x = expand(x1_,x2_,degree);
    % feature normalization
    x(:,2:end) = (x(:,2:end) - avg)./var;
    % evaluating hypothesis
    y = sigmoid(x*T);
    % plotting
    cAx = subplot(1,3,1);    
    v = mesh*T(:,1);
    gscatter(X(:,2),X(:,3),Y(:,1));
    hold(cAx,'on');
    contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
    xlabel('x_1');
    ylabel('x_2');    
    grid(cAx,'on');    
    title(cAx,'Positive Class (y=1): Setosa');
    plot(cAx,x1,x2,'kX','MarkerSize',10);
    hold(cAx,'off');
    legend(cAx,'0','1','Boundary',sprintf('p(y=1|x,%s) = %.2f','\theta',y(1)));        
    cAx = subplot(1,3,2);    
    v = mesh*T(:,2);
    gscatter(X(:,2),X(:,3),Y(:,2));
    hold(cAx,'on');
    contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
    xlabel('x_1');
    ylabel('x_2');
    grid(cAx,'on');
    title(cAx,'Positive Class (y=1): Versicolor');
    plot(cAx,x1,x2,'kX','MarkerSize',10);
    hold(cAx,'off');
    legend(cAx,'0','1','Boundary',sprintf('p(y=1|x,%s) = %.2f','\theta',y(2)));    
    cAx = subplot(1,3,3);
    v = mesh*T(:,3);
    gscatter(X(:,2),X(:,3),Y(:,3));
    hold(cAx,'on');
    contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
    xlabel('x_1');
    ylabel('x_2');
    grid(cAx,'on');
    title(cAx,'Positive Class (y=1): Virginica');
    plot(cAx,x1,x2,'kX','MarkerSize',10);
    hold(cAx,'off');
    legend(cAx,'0','1','Boundary',sprintf('p(y=1|x,%s) = %.2f','\theta',y(3)));
end    
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