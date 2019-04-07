 y_matrix = zeros(n,10); %one-versus-all algorthim trains N SVMs
    for label = 0:9
        idx =find(train_samples_labels == label);
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