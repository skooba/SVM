
function a = run_SVM(dataset,dataset_labels)
    %% Compute the kernel function
    [n,m] = size(dataset);
    sigma = sum(var(dataset));
    load K
    %K = gaussian_kernel(n,m,dataset,sigma);

    %% One-versus-rest algorithim 
    %Maximze the dual formulation of the SVM 
    y_matrix = zeros(n,10); %one-versus-all algorthim trains N SVMs
    for label = 0:9
        idx =find(dataset_labels == label);
        ONEvsREST = -ones(n,1);
        ONEvsREST(idx) = 1;
        a = SVM(n,ONEvsREST,K);
    %find value of threshold parameter
        support_vectors_index = find(a > .00001);
        [~,SV_length] = size(support_vectors_index');
        b = threshold(SV_length, ONEvsREST,a, support_vectors_index,K);
    %classify the vectors
        y = classify(n, support_vectors_index, ONEvsREST, a, K, b);
    %Matrix of outputs for each label
        y_matrix(:,label+1) = y;
    end
    %Get outputs for one-versus-all algorthim
    output = zeros(1,n);
    for k = 1:n
        maxi = max(y_matrix(k,:));
        output(k) = find(maxi == y_matrix(k,:))-1;
    end
    output = output';
    %% One-versus-one algorithim
    %initialize target vector with +1 and -1 for the two class SVM and 0 for
    %the other 8 classes
    output_matrixOvO = zeros(n,10); %one-versus-one algorthim trains N(N-1)/2 SVMS
    output_matrixOvO_delete = zeros(n,10); %will use for DAGSVM
    SVMnum = 0;
    skip = 0;
    column_names = [];
    for i = 0:9
        j = 0 + skip; %modify j so we do not repeat any SVMs
        skip = skip + 1;
        while j <= 9
            SVMnum = SVMnum + 1; %count the number SVM we are on
            idx_pos = find(dataset_labels == i);
            idx_neg = find(dataset_labels == j); 
            ONEvsONE = zeros(n,1);
            ONEvsONE(idx_pos) = 1;
            ONEvsONE(idx_neg) = -1;
            aOvO = SVM(n,ONEvsONE,K);
            support_vectors_indexOvO = find(aOvO > .00001);
            [~,SV_lengthOvO] = size(support_vectors_indexOvO');
            bOvO = threshold(SV_lengthOvO, ONEvsONE,aOvO, support_vectors_indexOvO,K);
            yOvO = classify(n, support_vectors_indexOvO, ONEvsONE, aOvO, K, bOvO);        
            %keep track of outputs for each iteration of algorthim
            idx_firstclass = find(yOvO >= 0);
            idx_secondclass = find(yOvO < 0);
            outputOvO = zeros(n,1);
            outputOvO(idx_firstclass) = i;
            outputOvO(idx_secondclass) = j;
            outputOvO_delete = zeros(n,1);
            outputOvO_delete(idx_firstclass) = j;
            outputOvO_delete(idx_secondclass) = i;
            output_matrixOvO(:,SVMnum) = outputOvO;
            output_matrixOvO_delete(:,SVMnum) = outputOvO_delete;
            column_names = [column_names, strcat(num2str(i),num2str(j))]; % keep track of column names for DAGSVM
            j = j + 1;
        end
    end
    %% One-versus-one algorithim
    %find the mode for each training point and assign the mode as the label
    output_class = mode(output_matrixOvO, 2);
    %% DAGSVM algorithim
    list = 0:9;
    matrix = repmat(list, [n,1]);
    [~, list_length] = size(matrix);
    while list_length > 1
        new_matrix = zeros(n, list_length-1);
        for row = 1:n
            row_values = matrix(row, :);
            find_column = strcat(num2str(row_values(1)),num2str(row_values(list_length)));
            column_number = round(strfind(column_names,find_column)/2);
            delete = output_matrixOvO_delete(row,column_number);
            new_row = row_values(row_values ~= delete);
            new_matrix(row, :) = new_row;
        end
        matrix = new_matrix;
        list_length = list_length - 1;
    end
    %% plot confusion matrices
    disp('The confusion matrix for the SVM one vs. all algorthim is:')
    conmat_ova = confusionmat(dataset_labels,output)
    disp('The accuracy for the SVM one vs. all algorthim is:')
    accuracy_ova = trace(conmat_ova)/n
    disp('The confusion matrix for the SVM one vs. one algorthim is:')
    conmat_ovo = confusionmat(dataset_labels,output_class)
    disp('The accuracy for the SVM one vs. one algorthim is:')
    accuracy_ovo = trace(conmat_ovo)/n
    disp('The confusion matrix for the DAGSVM algorthim is:')
    conmat_dagsvm = confusionmat(dataset_labels,matrix)
    disp('The accuracy for the DAGSVM algorthim is:')
    accuracy_dagsvm = trace(conmat_dagsvm)/n
end
