%% READING DATA
close all
load fisheriris
X = meas;
Y = strcmp(species,'versicolor');
[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
Y = Y(i);
%% TRAINING AND TEST SET DEFINITION
rng(10)
pct = 0.7;
m = size(X,1);
i = randperm(m)';
m = round(pct*m);
X = X(i,:);
Y = Y(i);
[U,~,L] = pca(X(1:m,:));
r = 2;
X = X*U(:,1:r);
fprintf('Variance retained: %.2f%%\n\n', (sum(L(1:r))/sum(L))*100);
degree = 3;
X = expand(X,degree);
n = size(X,2);
%% FEATURE SCALING AND MEAN NORMALIZATION
avg = mean(X(1:m,2:end));
var = std(X(1:m,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;
%% TRAINING CLASSIFIER
threshold = 0.5;
lambda = 0;
T = 1e-5 * rand(n,1);
options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
[T,~] = fminunc(@(T)(cost(T,X(1:m,:),Y(1:m),lambda)),T,options);
err = zeros(1,2); % 1 -> training set, 2 -> test
recall = zeros(1,2);
precision = zeros(1,2);
fscore = zeros(1,2);
J = zeros(1,2);
%% PLOTTING BOUNDARIES AND TEST RESULTS
beta = 1;
% metrics (train set)
J(1,1) = cost(T,X(1:m,:),Y(1:m),0);
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
err(1,1) = (fp+fn)/(tp+fp+tn+fn);
recall(1,1) = tp/(tp+fn);
precision(1,1) = tp/(tp+fp);
fscore(1,1) = (1+beta^2)*(precision(1,1).*recall(1,1))/((beta^2)*precision(1,1)+recall(1,1));
% metrics (test set)
J(1,2) = cost(T,X(m+1:end,:),Y(m+1:end),0);
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
err(1,2) = (fp+fn)/(tp+fp+tn+fn);
recall(1,2) = tp/(tp+fn);
precision(1,2) = tp/(tp+fp);
fscore(1,2) = (1+beta^2)*(precision(1,2).*recall(1,2))/((beta^2)*precision(1,2)+recall(1,2));
%% PRINTING SOME DATA
disp('----------------OVERALL RESULTS---------------------');
disp('--------------------TRAINING------------------------');
fprintf('Final Error J(Training): %.4f\n',J(1,1));
fprintf('Accuracy (1 - Err)(Training): %.2f\n',1-err(1,1));
fprintf('Recall (Training): %.2f\n',recall(1,1));
fprintf('Precision (Training): %.2f\n',precision(1,1));
fprintf('F%d Score (Training): %.2f\n',beta,fscore(1,1));
disp('---------------------TEST---------------------------');
fprintf('Final Error J(Test): %.4f\n',J(1,2));
fprintf('Accuracy (1 - Err)(Test): %.2f\n',1-err(1,2));
fprintf('Recall (Test): %.2f\n',recall(1,2));
fprintf('Precision (Test): %.2f\n',precision(1,2));
fprintf('F%d Score (Test): %.2f\n',beta,fscore(1,2));
disp('----------------------------------------------------');
%% FUNCTION TO COMPUTE COST AND GRADIENT
function [J, grad] = cost(T,X,y,lambda)
    m = size(X,1);
    h = sigmoid(X*T);
    J = -(1/m)*sum(y.*log(h)+(1-y).*log(1 - h)) + (lambda/(2*m))*sum(T(2:end).^2);
    grad = (1/m)*(X'*(h - y)) + [0; (lambda/m)*T(2:end)];
end
%% FUNCTION TO ADD POLYNOMIAL TERMS
function [X] = expand(X,degree)
    [m,n] = size(X);
    X = [X zeros(m,n*(degree-1))];
    for i = 2:degree
        X(:,(i-1)*n+1:i*n) = X(:,(i-2)*n+1:(i-1)*n) .* X(:,1:n);
    end
    X = [ones(m,1) X];
end
%% SIGMOID FUNCTION
function f = sigmoid (x)
    f = 1./(1+exp(-x));
end
%% INVERSE SIGMOID FUNCTION
function x = isigmoid (f)
    x = -log((1-f)/f);
end