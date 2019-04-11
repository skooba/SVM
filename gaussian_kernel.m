function K = gaussian_kernel(n1,n2,samples1,samples2,sigma)
K = zeros(n1,n2);
    for i = 1:n1
        i
        for j = 1:n2
            K(i,j) = exp((-(norm(samples1(i,:)-samples2(j,:)))^2)/(2*sigma^2));
        end
    end
end