function b = threshold(Ns, targets, a, index, K)
    sum2 = 0;
    for n = index'
        sum1 = 0;
        for m = index'
            sum1 = sum1 + a(m)*targets(m)*K(n,m);
        end
            sum2 = sum2 + targets(n) - sum1;
    end
b = sum2/Ns;
end