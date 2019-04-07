% instead of maximizing, minimize the negative of the objective function
function a = SVM(n,targets,K)
    H = zeros(n);
    for i=1:n
        for j=1:n
            H(i,j) = targets(i)*targets(j)*K(i,j);
        end
    end
    
    f = -ones(n,1);

    Aeq = targets.';
    beq = 0;
 
    lb = zeros(n,1);
    ub = ones(n,1);

    a = quadprog(H,f,[],[],Aeq,beq,lb,ub);
end