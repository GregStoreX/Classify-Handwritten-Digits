clc,clear;
disp('loading...');
load mnist_conv_masked; % features, r, masks
load training_testing_partition; % par, 5-fold
disp('load data finished');

r = double(nominal(r)); % transformed to nominal labels

for i=5:-1:1
    disp(['--cross validation ',num2str(i),' started--']);
    x_train = features(training(par,i),:);
    r_train = r(training(par,i));
    x_test = features(test(par,i),:);
    r_test = r(test(par,i));
    disp(['partition ',num2str(i),' finished']);
    
    tic;
    [B,dev,stats] = mnrfit(x_train,r_train);
    time_training = toc;
    disp('training finished');
    
    tic;
    pihat = mnrval(B,x_test); % 8400 * 10 matrix
    time_test = toc;
    disp('test finished');
    
    [max_val,y] = max(pihat'); % two 1 * 8400 matrices
    max_val = max_val';
    y = y';
    [conf_mat,order] = confusionmat(r_test,y);
    disp('confusion matrix generated');
    
    save(['cv_',num2str(i),'_results.mat'],'i','B','dev','stats','time_training','time_test','pihat','max_val','y','conf_mat','order');
    disp(['--cross validation ',num2str(i),' finished--']);
end
