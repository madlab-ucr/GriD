addpath(fullfile('~', 'tensor_toolbox'));

data_path = input('Enter the path to the file containing tensor data: ', 's');
data_directory = fileparts(data_path);
data_filename = strrep(data_path, '.mat', '');

% Load data from the .mat file
data = load(data_path);

% Extract the data and display tensor size
indices = data.indices;
% values = double(data.values(:)); % Column vector
tensor_size = data.size;

disp('Tensor size:');
disp(tensor_size);

% % Convert indices to 1-based indexing
% indices = indices + 1;

vals = ones(size(indices, 1), 1);

% Create the sparse tensor using sptensor
sparse_tensor = sptensor(indices, vals(:), permute(tensor_size, [3, 2, 1]));

% % Perform Tucker decomposition
% rank_tucker = input('Enter the Tucker decomposition rank: ');
% disp('Tucker decomposition...');
% [t_core, t_factors] = tucker_als(sparse_tensor, rank_tucker);
% % t_factors = hosvd(sparse_tensor, rank_tucker);
% tucker_factors_filename = [data_filename, '_tk_factors.mat'];
% tucker_factors_file = fullfile(data_directory, tucker_factors_filename);
% save(tucker_factors_file, 't_factors');

% Perform CP decomposition
rank_cp = input('Enter the CP decomposition rank: ');
disp('CP decomposition...');
cp_data = cp_als(sparse_tensor, rank_cp);
cp_factors_filename = [data_filename, '_cp_data_rank', num2str(rank_cp), '.mat'];
save(cp_factors_filename, 'cp_data', '-v7.3');
