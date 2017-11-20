%% READING DATA
iter = 1;
close all
load fisheriris
X = meas(:,[1 2]);
Y = strcmp(species,'versicolor');
[X,i] = unique(X,'rows');   % remove duplicates (care!: sorted results!!!)
Y = Y(i);
%% TRAINING AND TEST SET DEFINITION
% rng(10)
pct = 0.7;
m = size(X,1);
i = randperm(m)';
m = round(pct*m);
Y = Y(i);
%% TRYING DIFFERENT HYPHOTESIS COMPLEXITY BASED ON HOW MANY POLYNOMIAL FEATURES WE USE
threshold = 0.5;
lambda = 0.0;
beta = 1;
options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
D = 30;
err = zeros(D,2); % 1 -> training set, 2 -> test
recall = zeros(D,2);
precision = zeros(D,2);
fscore = zeros(D,12);
J = zeros(D,2);
for d = 1:D
    % building features
    x = expand(X(:,1),X(:,2),d);
    x = x(i,:);
    % feature scaling
    avg = mean(x(1:m,2:end));
    var = std(x(1:m,2:end));
    x(:,2:end) = (x(:,2:end) - avg)./var;
    % training classifier
    n = size(x,2);
    T = 1e-5 * rand(n,1);
    [T,~] = fminunc(@(T)(cost(T,x(1:m,:),Y(1:m),lambda)),T,options);
    J(d,1) = cost(T,x(1:m,:),Y(1:m),0);
    J(d,2) = cost(T,x(m+1:end,:),Y(m+1:end),0);
    % metrics (train set)
    h = sigmoid(x(1:m,:)*T);
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
    err(d,1) = (fp+fn)/(tp+fp+tn+fn);
    recall(d,1) = tp/(tp+fn);
    precision(d,1) = tp/(tp+fp);
    fscore(d,1) = (1+beta^2)*(precision(d,1).*recall(d,1))/((beta^2)*precision(d,1)+recall(d,1));
    % metrics (test set)
    h = sigmoid(x(m+1:end,:)*T);
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
    err(d,2) = (fp+fn)/(tp+fp+tn+fn);
    recall(d,2) = tp/(tp+fn);
    precision(d,2) = tp/(tp+fp);
    fscore(d,2) = (1+beta^2)*(precision(d,2).*recall(d,2))/((beta^2)*precision(d,2)+recall(d,2));
end
%% PLOTTING METRICS WITH RESPECT TO DEGREE
figure;
cAx = subplot(2,2,1);
hold(cAx,'on');
plot(cAx,(1:D),J(:,1),'ro');
plot(cAx,(1:D),J(:,2),'bo');
d = (1:0.1:D);
plot(d,spline((1:D),J(:,1),d),'r-','LineWidth',1.5);
plot(d,spline((1:D),J(:,2),d),'b-','LineWidth',1.5);
xlabel('Degree');
ylabel('Final Cost (Error)'); 
legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,2);
hold(cAx,'on');
plot(cAx,(1:D),err(:,1),'ro');
plot(cAx,(1:D),err(:,2),'bo');
plot(d,spline((1:D),err(:,1),d),'r-','LineWidth',1.5);
plot(d,spline((1:D),err(:,2),d),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Degree');
ylabel('Misclassification Error'); 
legend('Err (Training)','Err (Test)');
grid(cAx,'on');
hold(cAx,'off');
cAx = subplot(2,2,3);
hold(cAx,'on');
plot(cAx,(1:D),1 - fscore(:,1),'ro');
plot(cAx,(1:D),1 - fscore(:,2),'bo');
plot(d,spline((1:D),1 - fscore(:,1),d),'r-','LineWidth',1.5);
plot(d,spline((1:D),1 - fscore(:,2),d),'b-','LineWidth',1.5);
ylim([0 1]);
xlabel('Degree');
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