%Deleted from ONPMachineLearning.m

%% TRAINING CLASSIFIER - use fminunc, see if increasing lambda improves model. It doesn't, since there is not overfitting
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
    [T,~] = fminunc(@(T)(cost(T,X(1:m,:),y(1:m),lambda(l))),T,options);
    J(l,1) = cost(T,X(1:m,:),y(1:m),0);
    J(l,2) = cost(T,X(m+1:end,:),y(m+1:end),0);
    % metrics (train set)
    h = sigmoid(X(1:m,:)*T);
    output = h;
    output(h>=threshold) = 1;
    output(h<threshold) = 0;
    tp = find(output==1 & y(1:m)==1);
    tp = numel(tp);
    fp = find(output==1 & y(1:m)==0);
    fp = numel(fp);
    tn = find(output==0 & y(1:m)==0);
    tn = numel(tn);
    fn = find(output==0 & y(1:m)==1);
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
    tp = find(output==1 & y(m+1:end)==1);
    tp = numel(tp);
    fp = find(output==1 & y(m+1:end)==0);
    fp = numel(fp);
    tn = find(output==0 & y(m+1:end)==0);
    tn = numel(tn);
    fn = find(output==0 & y(m+1:end)==1);
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

 %% TRYING DIFFERENT HYPHOTHESIS COMPLEXITY BASED ON HOW MANY POLYNOMIAL FEATURES WE USE - does not work yet
% threshold = 0.5;
% lambda = 0.0;
% beta = 1;
% options = optimoptions('fminunc','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',1000);
% D = 30;
% err = zeros(D,2); % 1 -> training set, 2 -> test
% recall = zeros(D,2);
% precision = zeros(D,2);
% fscore = zeros(D,12);
% J = zeros(D,2);
% for d = 1:D
%     % building features
%     x = expand(X(:,1),X(:,2),d);
%     x = x(i,:);
%     % feature scaling
%     avg = mean(x(1:m,2:end));
%     var = std(x(1:m,2:end));
%     x(:,2:end) = (x(:,2:end) - avg)./var;
%     % training classifier
%     n = size(x,2);
%     T = 1e-5 * rand(n,1);
%     [T,~] = fminunc(@(T)(cost(T,x(1:m,:),Y(1:m),lambda)),T,options);
%     J(d,1) = cost(T,x(1:m,:),Y(1:m),0);
%     J(d,2) = cost(T,x(m+1:end,:),Y(m+1:end),0);
%     % metrics (train set)
%     h = sigmoid(x(1:m,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & Y(1:m)==1);
%     tp = numel(tp);
%     fp = find(output==1 & Y(1:m)==0);
%     fp = numel(fp);
%     tn = find(output==0 & Y(1:m)==0);
%     tn = numel(tn);
%     fn = find(output==0 & Y(1:m)==1);
%     fn = numel(fn);
%     err(d,1) = (fp+fn)/(tp+fp+tn+fn);
%     recall(d,1) = tp/(tp+fn);
%     precision(d,1) = tp/(tp+fp);
%     fscore(d,1) = (1+beta^2)*(precision(d,1).*recall(d,1))/((beta^2)*precision(d,1)+recall(d,1));
%     % metrics (test set)
%     h = sigmoid(x(m+1:end,:)*T);
%     output = h;
%     output(h>=threshold) = 1;
%     output(h<threshold) = 0;
%     tp = find(output==1 & Y(m+1:end)==1);
%     tp = numel(tp);
%     fp = find(output==1 & Y(m+1:end)==0);
%     fp = numel(fp);
%     tn = find(output==0 & Y(m+1:end)==0);
%     tn = numel(tn);
%     fn = find(output==0 & Y(m+1:end)==1);
%     fn = numel(fn);
%     err(d,2) = (fp+fn)/(tp+fp+tn+fn);
%     recall(d,2) = tp/(tp+fn);
%     precision(d,2) = tp/(tp+fp);
%     fscore(d,2) = (1+beta^2)*(precision(d,2).*recall(d,2))/((beta^2)*precision(d,2)+recall(d,2));
% end
% %% PLOTTING METRICS WITH RESPECT TO DEGREE
% figure;
% cAx = subplot(2,2,1);
% hold(cAx,'on');
% plot(cAx,(1:D),J(:,1),'ro');
% plot(cAx,(1:D),J(:,2),'bo');
% d = (1:0.1:D);
% plot(d,spline((1:D),J(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),J(:,2),d),'b-','LineWidth',1.5);
% xlabel('Degree');
% ylabel('Final Cost (Error)'); 
% legend('J^~(\theta)(Training)','J_t^~(\theta)(Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,2);
% hold(cAx,'on');
% plot(cAx,(1:D),err(:,1),'ro');
% plot(cAx,(1:D),err(:,2),'bo');
% plot(d,spline((1:D),err(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),err(:,2),d),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Degree');
% ylabel('Misclassification Error'); 
% legend('Err (Training)','Err (Test)');
% grid(cAx,'on');
% hold(cAx,'off');
% cAx = subplot(2,2,3);
% hold(cAx,'on');
% plot(cAx,(1:D),1 - fscore(:,1),'ro');
% plot(cAx,(1:D),1 - fscore(:,2),'bo');
% plot(d,spline((1:D),1 - fscore(:,1),d),'r-','LineWidth',1.5);
% plot(d,spline((1:D),1 - fscore(:,2),d),'b-','LineWidth',1.5);
% ylim([0 1]);
% xlabel('Degree');
% ylabel('1 - (F Score)'); 
% legend('F Score (Training)','F Score (Test)');
% grid(cAx,'on');
% hold(cAx,'off');