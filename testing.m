[n_test,~] = size(test_samples)
%sigma = sum(var(test_samples))
K_test = gaussian_kernel(n,n_test,train_samples,test_samples,.23);
%%
y_test = zeros(n_test,1);
y_test_matrix = zeros(n_test,9);
for class_index = 1:10
    for i = 1:n_test
         y_test(i) = sum(a_matrix(:,class_index).*y_matrix(:,class_index).*K_test(:,i))+b_vector(class_index);
    end
    y_test_matrix(:,class_index) = y_test;
end
%Get outputs for one-versus-all algorthim
output = zeros(1,n_test);
for k = 1:n_test
    maxi = max(y_test_matrix(k,:));
    output(k) = find(maxi == y_test_matrix(k,:))-1;
end
output = output';
%%
%find the y matrix for 45 test SVMs
y_OvO_test = zeros(n_test,1);
y_OvO_test_matrix = zeros(n_test,45);
for SVM_index = 1:45
    for i = 1:n_test
        y_OvO_test(i) = sum(aOvO_matrix(:,SVM_index).*yOvO_matrix(:,SVM_index).*K_test(:,i))+bOvO_vector(SVM_index);
    end
    y_OvO_test_matrix(:,SVM_index) = y_OvO_test;
end

%One vs One voting scheme
output_testOvO_matrix = zeros(n_test,45);
SVMnum = 0;
skip = 0;
column_names = [];
for i = 0:9
    j = 1 + skip; %modify j so we do not repeat any SVMs
    skip = skip + 1;
    while j <= 9
        SVMnum = SVMnum + 1;
        idx_firstclass = find(y_OvO_test_matrix(:,SVMnum) >= 0);
        idx_secondclass = find(y_OvO_test_matrix(:,SVMnum) < 0);
        output_testOvO = zeros(n_test,1);
        output_testOvO(idx_firstclass) = i;
        output_testOvO(idx_secondclass) = j;
        output_testOvO_matrix(:,SVMnum) = output_testOvO; 
        column_names = [column_names,strcat(num2str(i),num2str(j))];
        j = j + 1;
    end
end

output_OvO = mode(output_testOvO_matrix, 2);
%% DAGSVM algorithim
output_testmatrixOvO_delete = zeros(n_test,45);
skip = 0;
SVMnum = 0;
for i = 0:9
    j = 1 + skip; %modify j so we do not repeat any SVMs
    skip = skip + 1;
    while j <= 9;
        SVMnum = SVMnum + 1;
        idx_firstclass = find(y_OvO_test_matrix(:,SVMnum) >= 0);
        idx_secondclass = find(y_OvO_test_matrix(:,SVMnum) < 0);
        output_testOvO_delete = zeros(n_test,1);
        output_testOvO_delete(idx_firstclass) = j;
        output_testOvO_delete(idx_secondclass) = i;
        output_testmatrixOvO_delete(:,SVMnum) = output_testOvO_delete;
        j = j + 1;
    end
end

list = 0:9;
matrix = repmat(list, [n_test,1]);
[~, list_length] = size(matrix);
while list_length > 1
    new_matrix = zeros(n_test, list_length-1);
    for row = 1:n_test
        row_values = matrix(row, :);
        find_column = strcat(num2str(row_values(1)),num2str(row_values(list_length)));
        column_number = round(strfind(column_names,find_column)/2);
        delete = output_testmatrixOvO_delete(row,column_number);
        new_row = row_values(row_values ~= delete);
        new_matrix(row, :) = new_row;
    end
    matrix = new_matrix;
    list_length = list_length - 1;
end

%%
disp('The confusion matrix for the SVM one vs. all algorthim is:')
conmat_ova = confusionmat(test_samples_labels,output)
disp('The accuracy for the SVM one vs. all algorthim is:')
accuracy_ova = trace(conmat_ova)/n_test
disp('The confusion matrix for the SVM one vs. one algorthim is:')
conmat_ovo = confusionmat(test_samples_labels,output_OvO)
disp('The accuracy for the SVM one vs. one algorthim is:')
accuracy_ovo = trace(conmat_ovo)/n_test
disp('The confusion matrix for the DAGSVM algorthim is:')
conmat_dagsvm = confusionmat(test_samples_labels,matrix)
disp('The accuracy for the DAGSVM algorthim is:')
accuracy_dagsvm = trace(conmat_dagsvm)/n_test
