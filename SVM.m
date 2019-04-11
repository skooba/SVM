% instead of maximizing, minimize the negative of the objective function
function a = SVM(n,targets,K, C)
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
    ub = C*ones(n,1);
    
    %use soft margin for one vs all
    %a = quadprog(H,f,[],[],Aeq,beq,lb); 
    
    %use hard margin for one vs one
    a = quadprog(H,f,[],[],Aeq,beq,lb,ub);
end