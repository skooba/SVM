function K = gaussian_kernel(n,m,samples1,samples2,sigma)
K = zeros(n,n);
    for i = 1:n
        i
        for j = 1:m
            K(i,j) = exp((-(norm(samples1(i,:)-samples2(j,:)))^2)/(2*sigma^2));
        end
    end
end