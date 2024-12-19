function class = RSR(Q, dictionaries, threshold, k, classes)
    % Implementation of RSR based on pseudocode in section 4.5
    
    % Initialize variables
    inliers = 0;
    q = size(Q, 2);
    s = round(0.01*q);
    P = 0.99;
    class = classes(1);
    
    omega = (s + inliers) / q;
    lambda = log10(1-P) / log10(1-(omega^s));
    
    m = 0;
    while (lambda > 0 && m < 10000)
        % Choose s random descriptors from Q
        columns = [];
        i = 0;
        while i < s
            col = randi(size(Q, 2));
            if ~ismember(col, columns(:))
                columns = [columns col];
                i = i + 1;
            end
        end
        Qs = Q(:,columns);
               
        % Estimate class of Qs
        min_error = realmax;
        estimated_class = "";
        estimated_dictionary = [];
        for i=1:length(classes)
            D_cell = dictionaries(i);
            D = D_cell{1};
            
            X = [];
            for j=1:size(Qs, 2)
                col = Qs(:,j);
                Xi = omp(col, D, k);
                X = [X Xi];
            end

            error = norm(Qs - D*X)^2;
            
            if error < min_error
                estimated_class = classes(i);
                estimated_dictionary = D;
                min_error = error;
            end
        end
        
        % For every descriptor not in Qs
        new_inliers = 0;
        max_error = 0;
        for i=1:size(Q, 2)
            if ~(ismember(i, columns(:)))
                qi = Q(:,i);
                
                xs = [];
                for j=1:size(qi, 2)
                    col = qi(:,j);
                    xsi = omp(col, estimated_dictionary, k);
                    xs = [xs xsi];
                end
                
                error = norm(qi - estimated_dictionary*xs);
                if error > max_error
                    max_error = error;
                end

                if error <= threshold
                    new_inliers = new_inliers + 1;
                end
            end
        end
        
        % Update number of inliers
        if new_inliers > inliers
            inliers = new_inliers;
            class = estimated_class;
            omega = (s + inliers) / q;
            lambda = log(1-P) / log(1-(omega^s));
        end
        m = m + 1;
    end
end

