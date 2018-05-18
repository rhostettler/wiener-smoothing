% Two-Filter Bootstrap Particle Smoothing for Wiener State-Space Models
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
% 2015-03-20 -- Roland Hostettler <roland.hostettler@ltu.se>
% 2018-05-18 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;

%% Parameters
N = 100;    % No. of time samples
K = 20;     % No. of MC simulations
M = 500;    % No. of particles

%% Model
A = [
    1, 0.5;
    0, 1
];
Q = eye(2);
Lq = chol(Q).';
C = [1, 0];
g = @(x, ~) C*x;
m0 = [5; 3];
P0 = diag([10, 20]);
R = 1;
Lr = chol(R).';

model = model_wiener_ssm(A, Q, g, R, m0, P0);

%% Simulation
% Preallocate
x = zeros([2, N+1, K]);
y = zeros([1, N+1, K]);
xhat_2bfs = zeros([2, N, K]);
xhat_rtss = zeros([2, N, K]);
t = 1:N;

fh = pbar(K);
for k = 1:K
    %% Generate data
    x(:, 1, k) = m0 + chol(P0).'*randn([2, 1]);
    for n = 2:N+1
        x(:, n, k) = A*x(:, n-1, k) + Lq*randn([2, 1]);
        y(:, n, k) = C*x(:, n, k) + Lr*randn([1, 1]);
    end
    
    %% Estimation
    % 2F-PS
    tmp = wiener_2bfs(y(:, 2:N+1, k), t, model, M);
    xhat_2bfs(:, :, k) = tmp(:, 2:end);
    
    % RTSS
    [ms, Ps] = rtss(y(:, 2:N+1, k), A, C, Q, R, m0, P0);
    xhat_rtss(:, :, k) = squeeze(ms);
    
    %% Update
    pbar(k, fh);
end
pbar(0, fh);

% Remove initial states from the states and measurements
x = x(:, 2:N+1, :);
y = y(:, 2:N+1, :);

%% Calculate the Error
e_tfps = x-xhat_2bfs;
e_rtss = x-xhat_rtss;

%% Illustrations
figure(1); clf();
subplot(211);
plot([mean(e_tfps(1, :, :), 3).', mean(e_rtss(1, :, :), 3).']);
legend('2BFS', 'RTSS');
title('Mean Error');
subplot(212);
plot([mean(e_tfps(2, :, :), 3).', mean(e_rtss(2, :, :), 3).']);

figure(2); clf();
subplot(211);
semilogy([var(e_tfps(1, :, :), [], 3).', var(e_rtss(1, :, :), [], 3).']);
legend('2BFS', 'RTSS');
title('Mean Squared Error');
subplot(212);
semilogy([var(e_tfps(2, :, :), [], 3)', var(e_rtss(2, :, :), [], 3).']);
