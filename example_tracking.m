% Two-Filter Bootstrap Particle Smoothing for Wiener State-Space Models
%
% This is the second example in [1] where the two-filter particle smoother
% is applied to a bearings-only tracking problem.
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
N = 40;     % No. of time samples
K = 10;     % No. of Monte Carlo simulations
M = 500;    % No. of particles
Ts = 1;     % Sampling time

%% Model
% Initial state
m0 = [-10; 25; 2; -1];
P0 = diag([10, 5, 1, 1]);

% Dynamic model
F = [1, 0, Ts,  0; ...
     0, 1,  0, Ts; ...
     0, 0,  1,  0; ...
     0, 0,  0,  1];
Q = eye(4);

% Sensor model
g = @(x, ~) [sqrt(x(1, :).^2 + x(2, :).^2); atan2(x(2, :), x(1, :))];
R = diag([1 0.01]);

% Complete model
model = model_wiener_ssm(F, Q, g, R, m0, P0);

%% Smoothing parameters
par = struct();
par.Mt = round(M/3);

%% Simulation
% Preallocate
x = zeros([4, N+1, K]);
y = zeros([2, N+1, K]);
L0 = chol(P0).';
Lq = chol(Q).';
Lr = chol(R).';
for k = 1:K
    x(:, 1, k) = m0 + L0*randn([4, 1]);
    for n = 2:N+1
        x(:, n, k) = F*x(:, n-1, k) + Lq*randn([size(Q, 1), 1]);
        y(:, n, k) = g(x(:, n, k)) + Lr*randn([size(R, 1), 1]);
    end
end
t = (1:N)*Ts;
x = x(:, 2:N+1, :);
y = y(:, 2:N+1, :);

%% Estimation
xhat_2bfs = zeros([4, N, K]);
xhat_ffbsi = zeros([4, N, K]);
t_tfps = zeros(1, K);
t_ffbsi = zeros(1, K);
fh = pbar(K);
for k = 1:K
    % Two-filter smoother
    tstart = tic();
    tmp = wiener_2bfs(y(:, :, k), t, model, M, par);
    xhat_2bfs(:, :, k) = tmp(:, 2:end, :);
    t_tfps(:, k) = toc(tstart);

    % FFBSi
    tstart = tic();
    xhat_ffbsi(:, :, k) = ffbsi_ps(y(:, :, k), t, model, 2*M, M, par);
    t_ffbsi(:, k) = toc(tstart);
    
    % Progress update
    pbar(k, fh);
end
pbar(0, fh);

%% Error
e_tfps = x-xhat_2bfs;
e_ffbsi = x-xhat_ffbsi;
mse_tfps = [mean(e_tfps(1, :, :).^2 + e_tfps(2, :, :).^2, 3); ...
            mean(e_tfps(3, :, :).^2 + e_tfps(4, :, :).^2, 3)];
mse_ffbsi = [mean(e_ffbsi(1, :, :).^2 + e_ffbsi(2, :, :).^2, 3); ...
             mean(e_ffbsi(3, :, :).^2 + e_ffbsi(4, :, :).^2, 3)];
         
%% Illustrations
iExample = K;

figure(1); clf();
plot( ...
    [x(1, :, iExample).', xhat_2bfs(1, :, iExample).', xhat_ffbsi(1, :, iExample).'], ...
    [x(2, :, iExample).', xhat_2bfs(2, :, iExample).', xhat_ffbsi(2, :, iExample).'] ...
);
legend('Trajectory', '2BFS', 'FFBSi');

figure(2); clf();
subplot(211);
plot(y(1, :, iExample));
title('Range');
subplot(212);
plot(y(2, :, iExample));
title('Bearing');
 
figure(3); clf();
for i = 1:4
    subplot(4, 1, i);
    plot([mean(e_tfps(i, :, :), 3).', mean(e_ffbsi(i, :, :), 3).']);
    legend('2BFS', 'FFBSi');
    title('Mean Error');
end

figure(4); clf();
for i = 1:4
    subplot(4, 1, i);
    semilogy([var(e_tfps(i, :, :), [], 3).', var(e_ffbsi(i, :, :), [], 3).']);
    legend('2BFS', 'FFBSi');
    title('Mean Squared Error');
end

figure(5); clf();
for i = 1:2
    subplot(2, 1, i);
    semilogy([mse_tfps(i, :).', mse_ffbsi(i, :).'])
    legend('2BFS', 'FFBSi');
    title('Mean Squared Error (Position and Speed)');
end
