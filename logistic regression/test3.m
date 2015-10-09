clc,clear;
load mnist
load training_testing_partition
tic;
x_train = x(training(par,3),:);
[coeff,scores,latent]=pca(x_train');
e = scores(:,1:64)';
transformed = e*x';
transformed = transformed';
time_pca = toc;
r = double(nominal(r));
x_train = transformed(training(par,3),:);
x_test = transformed(test(par,3),:);
r_test = r(test(par,3),:);
r_train = r(training(par,3),:);
tic;[B dev stats]=mnrfit(x_train,r_train);time_training=toc;
tic;pihat=mnrval(B,x_test);time_test=toc;
[max_val,y]=max(pihat');
max_val = max_val';
y=y';
[conf_mat,order]=confusionmat(r_test,y);
sum(diag(conf_mat))/sum(sum(conf_mat))
save('pca_f64_cv_3_results.mat','B','dev','stats','time_pca','time_training','time_test','pihat','max_val','y','conf_mat','order');