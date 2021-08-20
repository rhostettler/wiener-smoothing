% Two filter bootstrap particle smoothing for Wiener state-space models
%
% This is the first example in [1] where the performance of the two-filter
% particle smoother for Wiener state-space models is illustrated. The
% smoother is compared to the Rauch-Tung-Striebel smoother in order to 
% illustrate the performance.
% 
% [1] R. Hostettler, "A two filter particle smoother for Wiener state-space
%     systems," in IEEE Conference on Control Applications (CCA), Sydney, 
%     Australia, September 2015.
%
% 2015-present -- Roland Hostettler

% Housekeeping
clear variables;
addpath lib;

%% Parameters
N = 100;    % No. of time samples
K = 100;     % No. of MC simulations
J = 500;    % No. of particles

%% Model
F = [
    1, 0.5;
    0, 1
];
Q = eye(2);
Lq = chol(Q).';
C = [1, 0];
g = @(x, ~) C*x;
Gx = @(~, ~) C;
m0 = [5; 3];
P0 = diag([10, 20]);
R = 1;
Lr = chol(R).';
model = model_wiener(F, Q, g, Gx, R, m0, P0);
model.py.fast = true;

%% Simulation
% Preallocate
x = zeros([2, N, K]);
y = zeros([1, N, K]);
xhat_bfps = zeros([2, N, K]);
xhat_rtss = zeros([2, N, K]);

fh = pbar(K);
for k = 1:K
    %% Generate data
    [x(:, :, k), y(:, :, k)] = simulate_model(model, [], N);
    
    %% Estimation
    % 2F-PS
    xhat_bfps(:, :, k) = wiener_bfps(model, y(:, :, k), [], J);

    % RTSS
    [ms, Ps] = rtss(y(:, :, k), F, C, Q, R, m0, P0);
    xhat_rtss(:, :, k) = squeeze(ms);
    
    %% Update
    pbar(k, fh);
end
pbar(0, fh);

%% Calculate the Error
e_tfps = x-xhat_bfps;
e_rtss = x-xhat_rtss;

%% Illustrations
figure(1); clf();
subplot(211);
plot([mean(e_tfps(1, :, :), 3).', mean(e_rtss(1, :, :), 3).']);
legend('BFPS', 'RTSS');
title('Mean Error');
subplot(212);
plot([mean(e_tfps(2, :, :), 3).', mean(e_rtss(2, :, :), 3).']);

figure(2); clf();
subplot(211);
plot([var(e_tfps(1, :, :), [], 3).', var(e_rtss(1, :, :), [], 3).']);
legend('BFPS', 'RTSS');
title('Mean Squared Error');
subplot(212);
plot([var(e_tfps(2, :, :), [], 3)', var(e_rtss(2, :, :), [], 3).']);
