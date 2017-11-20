%% SETUP
close all
load fisheriris
y = strcmp(species,'versicolor');
m = size(meas,1);
maxiter = 10000;
mindJ = 1e-5;
%% LOGISTIC REGRESSION CLASSIFIER WITH ONE FEATURE
% data and params
X = [ones(m,1) meas(:,3)];
T = 1e-4*rand(2,1);
% figure
fig = figure('KeyPressFcn', @MyKeyDown);
handles = guidata(fig); 
handles.stopnow = false;
guidata(fig,handles);
% gradient params
alpha = 1;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X*T);
niter = 1;
J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
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
        gAx = subplot(2,1,1);
        gscatter(X(:,2),y,y);    
        hold(gAx,'on')
        plot([-T(1)/T(2) -T(1)/T(2)],[0 1],'m-','LineWidth',2);
        xlabel('Petal Length (x_1)');
        ylabel('Is Versicolor? (Yes = 1/No = 0)');
        legend('Not Versicolor','Versicolor','Boundary');
        hold(gAx,'off');
        grid(gAx,'on');
        drawnow;
    end
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X'*(h - y));
    T = T - alpha * grad;
    h = sigmoid(X*T);
    J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
    dJ = J(niter - 1) - J(niter);
end
% final plot
jAx = subplot(2,1,2);
plot(jAx,1:niter,J(1:niter),'k.-');
xlabel('Iteration');
ylabel('J(\theta)');
legend('J(\theta)');
grid(jAx,'on');
gAx = subplot(2,1,1);
gscatter(X(:,2),y,y);    
hold(gAx,'on')
plot([-T(1)/T(2) -T(1)/T(2)],[0 1],'m-','LineWidth',2);
xlabel('Petal Length (x_1)');
ylabel('Is Versicolor? (Yes = 1/No = 0)');
legend('Not Versicolor','Versicolor','Boundary');
hold(gAx,'off');
grid(gAx,'on');
% display params
niter
T'
output = sigmoid(X*T);
output(output>=0.5) = 1;
output(output<0.5) = 0;
missed = sum(abs(output-y))
waitforbuttonpress;
%% LOGISTIC REGRESSION CLASSIFIER WITH TWO FEATURES
% data and params
X = [ones(m,1) meas(:,1) meas(:,2)];
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
alpha = 0.3;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X*T);
niter = 1;
J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
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
        gAx = subplot(2,1,1);
        gscatter(X(:,2),X(:,3),y);    
        hold(gAx,'on')
        plot(xlim(gAx),(-T(1) - T(2)*xlim())./T(3),'m-','LineWidth',2);
        xlabel('Sepal Length (x_1)');
        ylabel('Sepal Width (x_2)');
        legend('Not Versicolor','Versicolor','Boundary');
        hold(gAx,'off');
        grid(gAx,'on');
        drawnow;
    end
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X'*(h - y));
    T = T - alpha * grad;
    h = sigmoid(X*T);
    J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
    dJ = J(niter - 1) - J(niter);
end
% final plot
jAx = subplot(2,1,2);
plot(jAx,1:niter,J(1:niter),'k.-');
xlabel('Iteration');
ylabel('J(\theta)');
legend('J(\theta)');
grid(jAx,'on');
gAx = subplot(2,1,1);
gscatter(X(:,2),X(:,3),y);    
hold(gAx,'on')
plot(xlim(gAx),(-T(1) - T(2)*xlim())./T(3),'m-','LineWidth',2);
xlabel('Sepal Length (x_1)');
ylabel('Sepal Width (x_2)');
legend('Not Versicolor','Versicolor','Boundary');
hold(gAx,'off');
grid(gAx,'on');
drawnow;
% display params
niter
T'
output = sigmoid(X*T);
output(output>=0.5) = 1;
output(output<0.5) = 0;
missed = sum(abs(output-y))
waitforbuttonpress;
%% LOGISTIC REGRESSION CLASSIFIER WITH TWO FEATURES ADDING SOME POLYNOMIAL FEATURES AS WELL
% data and params
X = [ones(m,1) meas(:,1) meas(:,2) meas(:,1).^3 meas(:,2).^3];
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
alpha = 0.3;
dJ = mindJ;
J = zeros(maxiter,1);
h = sigmoid(X*T);
niter = 1;
J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
% mesh building
P = 50;
mx = linspace(bottom(1),top(1),P);
my = linspace(bottom(2),top(2),P);
[mx,my] = meshgrid(mx,my);
mesh = [ones(numel(mx),1) mx(:) my(:) mx(:).^2 my(:).^3];
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
        xlabel('Sepal Length (x_1)');
        ylabel('Sepal Width (x_2)');
        legend('Not Versicolor','Versicolor','Boundary');
        hold(gAx,'off');
        grid(gAx,'on');
        drawnow;
    end
    % gradient step
    niter = niter + 1;
    grad = (1/m)*(X'*(h - y));
    T = T - alpha * grad;
    h = sigmoid(X*T);
    J(niter) = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h));
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
xlabel('Sepal Length (x_1)');
ylabel('Sepal Width (x_2)');
legend('Not Versicolor','Versicolor','Boundary');
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
%% SIGMOID FUNCTION
function f = sigmoid (x)
    f = 1./(1+exp(-x));
end