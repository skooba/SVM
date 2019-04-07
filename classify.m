function y = classify(n, index, targets, a, K, b)
    y = zeros(n,1);
    for i = index'
        y = y + a(i)*targets(i)*K(:,i);
    end
    y = y + b;
end