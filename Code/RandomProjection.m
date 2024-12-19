function new_matrix = RandomProjection(feature_matrix, n)
    R = randn(n, size(feature_matrix, 1));
    new_matrix = R*feature_matrix;
end

