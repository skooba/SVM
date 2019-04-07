  %% Compute the kernel function
    [n,m] = size(train_samples);
    %sigma = sum(var(train_samples));
    %load K
    K = gaussian_kernel(n,n,train_samples,train_samples,.05);

    %% One-versus-rest algorithim 
    %Maximze the dual formulation of the SVM 
    y_matrix = zeros(n,10); %one-versus-all algorthim trains N SVMs
    a_matrix = zeros(n,10);
    b_vector = zeros(10,1);
    for label = 0:9
        idx =find(train_samples_labels == label);
        ONEvsREST = -ones(n,1);
        ONEvsREST(idx) = 1;
        a = SVM(n,ONEvsREST,K);
        a_matrix(:,label+1) = a;
    %find value of threshold parameter
        support_vectors_index = find(a > .00001);
        [~,SV_length] = size(support_vectors_index');
        b = threshold(SV_length, ONEvsREST,a, support_vectors_index,K);
        b_vector(label+1) = b;
    %classify the vectors
        y = classify(n, support_vectors_index, ONEvsREST, a, K, b);
    %Matrix of outputs for each label
        y_matrix(:,label+1) = y;
    end
    %% One-versus-one algorithim
    %initialize target vector with +1 and -1 for the two class SVM and 0 for
    %the other 8 classes
%    output_matrixOvO = zeros(n,10); %one-versus-one algorthim trains N(N-1)/2 SVMS
%    output_matrixOvO_delete = zeros(n,10); %will use for DAGSVM
    SVMnum = 0;
    skip = 0;
    bOvO_vector = zeros(45,1)
    yOvO_matrix = zeros(n,45);
    aOvO_matrix = zeros(n,45);
    column_names = [];
    for i = 0:9
        j = 1 + skip; %modify j so we do not repeat any SVMs
        skip = skip + 1;
        while j <= 9
            SVMnum = SVMnum + 1; %count the number SVM we are on
            idx_pos = find(train_samples_labels == i);
            %length(idx_pos);
            idx_neg = find(train_samples_labels == j); 
            %length(idx_neg)
            ONEvsONE = zeros(n,1);
            ONEvsONE(idx_pos) = 1;
            ONEvsONE(idx_neg) = -1;
            aOvO = SVM(n,ONEvsONE,K);
            aOvO_matrix(:,SVMnum) = aOvO;
            support_vectors_indexOvO = find(aOvO > .0000001);
            [~,SV_lengthOvO] = size(support_vectors_indexOvO');
            bOvO = threshold(SV_lengthOvO, ONEvsONE,aOvO, support_vectors_indexOvO,K);
            bOvO_vector(SVMnum) = bOvO; 
            yOvO = classify(n, support_vectors_indexOvO, ONEvsONE, aOvO, K, bOvO);   
            yOvO_matrix(:,SVMnum) = yOvO;
            column_names = [column_names, i,j]; % keep track of column names for DAGSVM
            j = j + 1;
        end
    end
