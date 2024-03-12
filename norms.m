addpath(fullfile('~', 'tensor_toolbox'));

slice_path = input('Enter the path to the file containing test data: ', 's');

% Load data from the .mat file
slice_data = load(slice_path);

% Extract the data and display tensor size
slice_indices = double(slice_data.indices);
slice_tensor_size = slice_data.size;

disp('Test Tensor size:');
disp(slice_tensor_size);

slice_vals = ones(length(slice_indices), 1);

% Convert indices to 1-based indexing
% slice_indices = slice_indices + 1;

% Create the sparse tensor using sptensor
test_tensor = sptensor(slice_indices, slice_vals(:), slice_tensor_size);

% Ask the user for the location of the decomposition files
data = input('Enter the path to the file containing factor data: ', 's');

% Load the precomputed decomposition factors and core tensor
load(data)

U = cp_data.U;

% Initialize an array to store the fits of each slice
slice_fits = zeros(slice_tensor_size(1), 1);

disp('Calculating reconstruction error for each test document...')
% Loop over each slice and calculate the fit
for i = 1:slice_tensor_size(1)
    % Get the co-occurrence matrix for the current slice
    co_occurrence_matrix = test_tensor(i, :, :);
    
    % Calculate the reconstruction gs_r
    B = U{2};

    Xs = spmatrix(co_occurrence_matrix);

    part1 = sparse(pinv(B) * Xs);  
    part2 = sparse(part1*U{3});  
    part3 = sparse(part2* pinv(U{3}));  
    
    reconstruction = sparse(U{2} * part3);

    error = norm(sparse(reconstruction) - Xs, 'fro');

    fprintf("Error for slice %d: %f\n", i, error)

    slice_fits(i) = error;
end

save("tensor_data/reconstruction_errors_nogpt.mat", 'slice_fits', '-v7.3');
