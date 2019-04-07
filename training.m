    [n,m] = size(train_samples);
    sigma = sum(var(train_samples));
    load K
    %K = gaussian_kernel(n,m,train_samples,sigma);    
%% One-versus-rest algorithim 
    %Maximze the dual formulation of the SVM 
    y_matrix = zeros(n,10); %one-versus-all algorthim trains N SVMs
    weight_matrix = zeros(10,m)
    b_vector = zeros(10,1)
    for label = 0:9
        label
        idx =find(train_samples_labels == label);
        ONEvsREST = -ones(n,1);
        ONEvsREST(idx) = 1;
        a = SVM(n,ONEvsREST,K);
    %find value of threshold parameter
        support_vectors_index = find(a > .00001);
        [~,SV_length] = size(support_vectors_index');
        b = threshold(SV_length, ONEvsREST,a, support_vectors_index,K);
    %find the weights
        support_vectors = ONEvsREST(support_vectors_index)
        a_support_vectors = a(support_vectors_index)
        w = (a.*train_samples_labels).'*train_samples
        %w = zeros(n,1)
        %for i = 1:SV_length
        %w(support_vectors_index) = dot(a_support_vectors(i),support_vectors(i));
        %end
        weight_matrix(label+1,:) = w; %weights of N SVMs
        b_vector(label+1) = b
    end