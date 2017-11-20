%% SETUP
close all
load demolog1
m = size(X,1);
maxiter = 400;
mindJ = 1e-10;
%% LOGISTIC REGRESSION CLASSIFIER WITH TWO FEATURES
% data and params
degree = 10;
X = expand(X(:,1),X(:,2),degree);
bottom = min(X(:,[2,3]));
top = max(X(:,[2,3]));
avg = mean(X(:,2:end));
var = std(X(:,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;
T = 1e-4*rand(size(X,2),1);
% figure
fig = figure('KeyPressFcn', @MyKeyDown);
handles = guidata(fig); 
handles.stopnow = false;
guidata(fig,handles);
% gradient params
alpha = 2;
lambda = 0;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X*T);
niter = 1;
J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
% mesh building
P = 50;
mx = linspace(bottom(1),top(1),P);
my = linspace(bottom(2),top(2),P);
[mx,my] = meshgrid(mx,my);
mesh = expand(mx(:),my(:),degree);
mesh(:,2:end) = (mesh(:,2:end) - avg)./var;
% gradient descent
while (niter < maxiter && abs(dJ) >= mindJ)
    % step plot
    handles = guidata(fig);
    if (~handles.stopnow)
        jAx = subplot(2,1,2);
        plot(jAx,1:niter,J(1:niter),'k.-');
        xlabel('Iteration');
        ylabel('J(\theta)');
        legend('J(\theta)');
        grid(jAx,'on');
        v = mesh*T;
        gAx = subplot(2,1,1);
        gscatter(X(:,2),X(:,3),y);    
        hold(gAx,'on');    
        contour(gAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
        xlabel('x_1');
        ylabel('x_2');
        legend('0','1','Boundary');
        hold(gAx,'off');
        grid(gAx,'on');
        drawnow;
    end
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
    T = T - alpha * grad;
    h = sigmoid(X*T);
    J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    dJ = J(niter - 1) - J(niter);
end
% final plot
jAx = subplot(2,1,2);
plot(jAx,1:niter,J(1:niter),'k.-');
xlabel('Iteration');
ylabel('J(\theta)');
legend('J(\theta)');
grid(jAx,'on');
v = mesh*T;
gAx = subplot(2,1,1);
gscatter(X(:,2),X(:,3),y);    
hold(gAx,'on');    
contour(gAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),[0 0],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');
legend('0','1','Boundary');
hold(gAx,'off');
grid(gAx,'on');
% display params
niter
T'
output = sigmoid(X*T);
output(output>=0.5) = 1;
output(output<0.5) = 0;
missed = sum(abs(output-y))
%% FUNCTION TO HANDLE KEY PRESS
function MyKeyDown(hObject, ~, ~)
    handles = guihandles(hObject); 
    handles.stopnow = true;
    guidata(hObject,handles);
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