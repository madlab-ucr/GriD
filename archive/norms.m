addpath(fullfile('~', 'tensor_toolbox'));

data_path = input('Enter the path to the file containing tensor data: ', 's');

% Load data from the .mat file
data = load(data_path);

% Extract the data and display tensor size
indices = data.indices;
values = double(data.values(:)); % Column vector
tensor_size = data.size;

disp('Tensor size:');
disp(tensor_size);

% Convert indices to 1-based indexing
indices = indices + 1;

vals = ones(size(indices, 1), 1);

% Create the sparse tensor using sptensor
sparse_tensor = sptensor(indices, vals(:), tensor_size);

% Ask the user for the location of the decomposition files
data = input('Enter the path to the file containing core data: ', 's');

% Load the precomputed decomposition factors and core tensor
load(data)

N = ndims(sparse_tensor);
U = cp_data.U;
% U = c_core.U;

% Initialize an array to store the fits of each slice
slice_fits = zeros(tensor_size(1), 1);

% Loop over each slice and calculate the fit
for i = 1:tensor_size(1)
    % Extract the k-th slice from the tensor
    slice_tensor = sparse_tensor(i, :, :);
    
    %% Reconstruct the k-th slice using the factor matrices A, B, and C
    reconstructed_slice = U{3} * diag(cp_data.lambda) * diag(U{1}(i, :)) * U{2}';

    % Calculate the fit of the k-th slice (e.g., Frobenius norm)
    diff = slice_tensor - sptensor(reconstructed_slice);
    fit = norm(diff);

    fprintf("Fit for slice %d: %f\n", i, fit)
    
    % Store the fit of the k-th slice
    slice_fits(i) = fit;
end

save("tensor_data/reconstruction_errors.mat", 'slice_fits', '-v7.3');
