% The mathematical equation of the two-tailed test customized for this research [ZHENG2004]
% E1 is the error rate for classifier M1 E2 is the error rate for classifier M2
% q=(E1 +E2)2;
% n is the number of records in the testing data set.
% The value of T determines the statistical significance between the two algorithms M1 and M2.
function T = TwoTailedTest(E1, E2, n)

q = (E1+E2)/2;

T = abs(E1-E2)/sqrt(q*(1-q)*(2/n));
