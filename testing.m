% load MNIST
% load y_matrix
% load a
[n_test,m_test] = size(test_samples)
% K_test = gaussian_kernel(n_test,m_test,train_samples,.5)
%K_test = gaussian_kernel(4000,n_test,train_samples,test_samples,.5);
for i = 1:1000
    y_test(i) = sum(a(i)*y(i)*K_test(:,i))+b
end
% for i = 1:10
%     y = y_matrix(:,1)
%     y_test = a(i)*y(i)*K_test + b
%output = zeros(1,n);
%y = weight_matrix(1,:)*test_samples.' + b_vector(1)
%for k = 1:n
%    maxi = max(y_matrix(k,:));
%    output(k) = find(maxi == y_matrix(k,:))-1;
%end
%output = output';