%% READING DATA
iter = 1;
close all
load fisheriris
X = meas(:,[1 2]);
Y = strcmp(species,'versicolor');
[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
Y = Y(i);
degree = 20;
X = expand(X(:,1),X(:,2),degree);
%% TRAINING AND TEST SET DEFINITION
% rng(10)
pct = 0.7;
[m,n] = size(X);
i = randperm(m)';
m = round(pct*m);
X = X(i,:);
Y = Y(i);
%% FEATURE SCALING AND MEAN NORMALIZATION
avg = mean(X(1:m,2:end));
var = std(X(1:m,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;
%% TRAINING CLASSIFIER
L = 12;
lambda = [0 0.01*(2.^(0:L-2))];
threshold = 0.5;
beta = 1;
options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
err = zeros(L,2); % 1 -> training set, 2 -> test
recall = zeros(L,2);
precision = zeros(L,2);
fscore = zeros(L,2);
J = zeros(L,2);
for l = 1:L
    % training classifier
    n = size(X,2);
    T = 1e-5 * rand(n,1);
    [T,~] = fminunc(@(T)(cost(T,X(1:m,:),Y(1:m),lambda(l))),T,options);
    J(l,1) = cost(T,X(1:m,:),Y(1:m),0);
    J(l,2) = cost(T,X(m+1:end,:),Y(m+1:end),0);
    % metrics (train set)
    h = sigmoid(X(1:m,:)*T);
    output = h;
    output(h>=threshold) = 1;
    output(h<threshold) = 0;
    tp = find(output==1 & Y(1:m)==1);
    tp = numel(tp);
    fp = find(output==1 & Y(1:m)==0);
    fp = numel(fp);
    tn = find(output==0 & Y(1:m)==0);
    tn = numel(tn);
    fn = find(output==0 & Y(1:m)==1);
    fn = numel(fn);
    err(l,1) = (fp+fn)/(tp+fp+tn+fn);
    recall(l,1) = tp/(tp+fn);
    precision(l,1) = tp/(tp+fp);
    fscore(l,1) = (1+beta^2)*(precision(l,1).*recall(l,1))/((beta^2)*precision(l,1)+recall(l,1));
    % metrics (test set)
    h = sigmoid(X(m+1:end,:)*T);
    output = h;
    output(h>=threshold) = 1;
    output(h<threshold) = 0;
    tp = find(output==1 & Y(m+1:end)==1);
    tp = numel(tp);
    fp = find(output==1 & Y(m+1:end)==0);
    fp = numel(fp);
    tn = find(output==0 & Y(m+1:end)==0);
    tn = numel(tn);
    fn = find(output==0 & Y(m+1:end)==1);
    fn = numel(fn);
    err(l,2) = (fp+fn)/(tp+fp+tn+fn);
    recall(l,2) = tp/(tp+fn);
    precision(l,2) = tp/(tp+fp);
    fscore(l,2) = (1+beta^2)*(precision(l,2).*recall(l,2))/((beta^2)*precision(l,2)+recall(l,2));
end
%% PLOTTING METRICS WITH RESPECT TO LAMBDA
figure;
cAx = subplot(2,2,1);
hold(cAx,'on');
plot(cAx,lambda,J(:,1),'ro');
plot(cAx,lambda,J(:,2),'bo');
l = (0:0.1:L);
plot(l,spline(lambda,J(:,1),l),'r-','LineWidth',1.5);
plot(l,spline(lambda,J(:,2),l),'b-','LineWidth',1.5);
xlabel('Lambda');
ylabel('Final Cost (Error)'); 
legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,2);
hold(cAx,'on');
plot(cAx,lambda,err(:,1),'ro');
plot(cAx,lambda,err(:,2),'bo');
plot(l,spline(lambda,err(:,1),l),'r-','LineWidth',1.5);
plot(l,spline(lambda,err(:,2),l),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Lambda');
ylabel('Misclassification Error'); 
legend('Err (Training)','Err (Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,3);
hold(cAx,'on');
plot(cAx,lambda,1 - fscore(:,1),'ro');
plot(cAx,lambda,1 - fscore(:,2),'bo');
plot(l,spline(lambda,1 - fscore(:,1),l),'r-','LineWidth',1.5);
plot(l,spline(lambda,1 - fscore(:,2),l),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Lambda');
ylabel('1 - (F Score)'); 
legend('F Score (Training)','F Score (Test)');
grid(cAx,'on');
hold(cAx,'off');
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
%% INVERSE SIGMOID FUNCTION
function x = isigmoid (f)
    x = -log((1-f)/f);
end