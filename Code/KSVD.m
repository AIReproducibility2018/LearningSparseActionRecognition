function D = KSVD(initD, X, noIt, s)
    D = initD;
    atoms = size(D, 2);
    for i=1:noIt
        W = [];
        for j=1:size(X, 2)
            col = X(:,j);
            Wi = omp(col, D, s);
            W = [W Wi];
        end   
        for j=1:5
            mE = X - D*W;
            for k = 1:atoms
                vP = find(W(k,:));
                if ~isempty(vP)
                    mEP = mE(:, vP) + D(:,k) * W(k, vP);
                    vA = D(:,k)' * mEP;
                    W(k,vP) = vA;
                    col = mEP * vA' / (vA * vA');
                    D(:,k) = col;
                end
            end
        end
        D = bsxfun(@rdivide, D, sqrt(sum(D.^2, 1)));
    end
end
