%% READING DATA
iter = 1;
close all
load fisheriris
X = meas(:,[1 2]);
Y = strcmp(species,'versicolor');
[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
Y = Y(i);
degree = 6;
X = expand(X(:,1),X(:,2),degree);
%% TRAINING AND TEST SET DEFINITION
% rng(10)
pct = 0.7;
[m,n] = size(X);
i = randperm(m)';
m = round(pct*m);
% i1 = find(Y(i)==1); i0 = find(Y(i)==0); p = round(pct*numel(i0)); i = [i(i0(1:p)); i(i1(2:end)); i(i0(p+1:end)); i(i1(1))]; m = m - (numel(i0) - p + 1);
X = X(i,:);
Y = Y(i);
%% FEATURE SCALING AND MEAN NORMALIZATION
bottom = min(X(:,[2,3]));
top = max(X(:,[2,3]));
avg = mean(X(1:m,2:end));
var = std(X(1:m,2:end));
X(:,2:end) = (X(:,2:end) - avg)./var;
%% MESH DEFINITION
P = 50;
mx = linspace(bottom(1),top(1),P);
my = linspace(bottom(2),top(2),P);
[mx,my] = meshgrid(mx,my);
mesh = expand(mx(:),my(:),degree);
mesh(:,2:end) = (mesh(:,2:end) - avg)./var;
%% TRAINING CLASSIFIER
threshold = 0.5;
lambda = 0;
T = 1e-5 * rand(n,1);
trnJ = cost(T,X(1:m,:),Y(1:m),0);
tstJ = cost(T,X(m+1:end,:),Y(m+1:end),0);
%% PLOTTING BOUNDARIES AND TEST RESULTS
figure;
cAx = gca;    
v = mesh*T;
gscatter(X(1:m,2),X(1:m,3),Y(1:m));
hold(cAx,'on');
contour(cAx,reshape(mesh(:,2),[P P]),reshape(mesh(:,3),[P P]),reshape(v,[P P]),isigmoid(threshold)*[1 1],'m','LineWidth',2);
xlabel('x_1');
ylabel('x_2');    
grid(cAx,'on');    
title(cAx,'Positive Class (y=1): versicolor');
h = sigmoid(X(m+1:end,:)*T);
output = h;
output(h>=threshold) = 1;
output(h<threshold) = 0;
% output(:) = 0;  % everything taste like chicken kind of classifier!!!
tp = find(output==1 & Y(m+1:end)==1);
plot(cAx,X(m+tp,2),X(m+tp,3),'k.','MarkerSize',12);
tp = numel(tp);
fp = find(output==1 & Y(m+1:end)==0);
plot(cAx,X(m+fp,2),X(m+fp,3),'kx','MarkerSize',10);
fp = numel(fp);
tn = find(output==0 & Y(m+1:end)==0);
plot(cAx,X(m+tn,2),X(m+tn,3),'k.','MarkerSize',12);
tn = numel(tn);
fn = find(output==0 & Y(m+1:end)==1);
plot(cAx,X(m+fn,2),X(m+fn,3),'kx','MarkerSize',10);
fn = numel(fn);
err = (fp+fn)/(tp+fp+tn+fn);
recall = tp/(tp+fn);
precision = tp/(tp+fp);
beta = 1;
fscore = (1+beta^2)*(precision.*recall)/((beta^2)*precision+recall);
hold(cAx,'off');
if (err > 0)
    legend(cAx,'0','1','Boundary','Correct','Incorrect');
else
    legend(cAx,'0','1','Boundary','Correct');
end
%% PRINTING SOME DATA
disp('--------------------RESULTS-------------------------');
fprintf('Final Error J(Training): %.4f\n',trnJ);
fprintf('Final Error J(Test): %.4f\n',tstJ);
fprintf('Accuracy (1 - Err)(Test): %.2f\n',1-err);
fprintf('Recall (Test): %.2f\n',recall);
fprintf('Precision (Test): %.2f\n',precision);
fprintf('F%d Score (Test): %.2f\n',beta,fscore);
disp('----------------------------------------------------');
%% PLOTTING METRICS WITH RESPECT TO THRESHOLD
P = 100; % number of points
threshold = linspace(0,1,P);
err = zeros(size(threshold));
recall = zeros(size(threshold));
precision = zeros(size(threshold));
fscore = zeros(size(threshold));
for i = 1:P
    output = h;
    output(h>=threshold(i)) = 1;
    output(h<threshold(i)) = 0;
    tp = find(output==1 & Y(m+1:end)==1);
    tp = numel(tp);
    fp = find(output==1 & Y(m+1:end)==0);
    fp = numel(fp);
    tn = find(output==0 & Y(m+1:end)==0);
    tn = numel(tn);
    fn = find(output==0 & Y(m+1:end)==1);
    fn = numel(fn);
    err(i) = (fp+fn)/(tp+fp+tn+fn);
    recall(i) = tp/(tp+fn);
    precision(i) = tp/(tp+fp);
    fscore(i) = (1+beta^2)*(precision(i).*recall(i))/((beta^2)*precision(i)+recall(i));    
end
figure;
cAx = gca;
hold(cAx,'on');
plot(cAx,threshold,1-err);
plot(cAx,threshold,precision);
plot(cAx,threshold,recall);
plot(cAx,threshold,fscore,'k-','LineWidth',2);
xlabel('Threshold');
ylabel('Metric'); 
ylim([0 1]);
xlim([0 1]);
legend('Accurary','Precision','Recall','F Score');
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