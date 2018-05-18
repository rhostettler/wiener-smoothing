% Two-Filter Particle Smoothing with Linear State Dynamics
%
% This is the second example in [1] where the two-filter particle smoother
% is applied to a bearings-only tracking problem. The smoother is compared
% to 
% particle smoother with linear state dynamics (Wiener system) is
% illustrated. The smoother is compared to the Rauch-Tung-Striebel smoother
% in order to illustrate the performance.
% 
% 2015-03-20 -- Roland Hostettler <roland.hostettler@ltu.se>

% Housekeeping
clear variables;

%% Parameters
T = 40; % time samples
N = 20; % MC sims
M = 500;
Ts = 1; % Sampling time (Ts = 1 for results)

%% System Description
A = [1, 0, Ts,  0; ...
     0, 1,  0, Ts; ...
     0, 0,  1,  0; ...
     0, 0,  0,  1];

% For experimentation only
% Bsim = [Ts^2/2,      0; ...
%              0, Ts^2/2; ...
%             Ts,      0; ...
%              0,     Ts];
% Qsim = diag([10, 5]);
% B = eye(4);
% Q = diag(diag(Bsim*Qsim*Bsim'));
% Bsim = eye(4);
% Qsim = eye(4);

% B & Q for the results in the paper.
B = eye(4);
Q = eye(4);

% Sensor model and noise covariance
h = @(x) [sqrt(x(1, :).^2 + x(2, :).^2); atan2(x(2, :), x(1, :))];
R = diag([1 0.01]);

% Initial distribution
mu0 = [-10; 25; 2; -1];
Sigma0 = diag([10, 5, 1, 1]);

%% Models & Algorithm Parameters
% 2F-PS model
model_tfps = struct();
model_tfps.A = @(t) A;
model_tfps.Q = @(t) B*Q*B';
model_tfps.likelihood = @(y, x, t) mvnpdf(y.', (h(x)).', R.').';
model_tfps.mu0 = mu0;
model_tfps.Sigma0 = Sigma0;

% 2F-PS parameters
par_tfps.M = M;
par_tfps.M_T = round(M/3);

% FFBSi Model
model_ffbsi = struct();
model_ffbsi.px0.rng = @(M) mu0*ones([1, M]) + chol(Sigma0, 'lower')*randn([4, M]);
model_ffbsi.px.rng = @(x, t) A*x + chol(Q, 'lower')*randn([size(Q, 1), size(x, 2)]);
model_ffbsi.px.eval = @(xp, x, t) mvnpdf(xp.', (A*x).', Q.').';
model_ffbsi.py.eval = @(y, x, t) mvnpdf(y.', (h(x)).', R.').';
model_ffbsi.px.rho = model_ffbsi.px.eval(zeros([4, 1]), zeros([4, 1]), 0);

% FFBSi Parameters
par_ffbsi.Mf = 2*M;
par_ffbsi.M_T = round(M/3);
par_ffbsi.Ms = M;

%% System Simulation
% Preallocate
x = zeros([4, T+1, N]);
y = zeros([2, T+1, N]);
xhat_tfps = zeros([4, T, N]);
xhat_ffbsi = zeros([4, T, N]);
t_tfps = zeros(1, N);
t_ffbsi = zeros(1, N);

% eta(N);
for k = 1:N
    % Initialize
    x(:, 1, k) = mu0 + chol(Sigma0, 'lower')*randn([4, 1]);

    % Simulate the system
    for n = 2:T+1
        t = n-1;

        % Propagate the state
        x(:, n, k) = A*x(:, n-1, k) + B*chol(Q, 'lower')*randn([size(Q, 1), 1]);
%         x(:, n, k) = A*x(:, n-1, k) + Bsim*chol(Qsim, 'lower')*randn([size(Qsim, 1), 1]);

        % Measurement
        y(:, n, k) = h(x(:, n, k)) + chol(R, 'lower')*randn([size(R, 1), 1]);
    end
end

touter = tic();
parfor_progress(N);
parfor k = 1:N    
% for k = 1:N
    %% Smoothing
    % 2F-PS
    tstart = tic();
    [xhat_tfps(:, :, k), sys] = tfps(y(:, 2:T+1, k), model_tfps, par_tfps);
    t_tfps(:, k) = toc(tstart);

    % FFBSi
    tstart = tic();
    xhat_ffbsi(:, :, k) = ffbsi_ps(y(:, 2:T+1, k), model_ffbsi, par_ffbsi);
    t_ffbsi(:, k) = toc(tstart);

    %% Progress Info
    parfor_progress();
    pause(10); % Cooldown
end
parfor_progress(0);
toc(touter);

% Remove initial states from the states and measurements
x = x(:, 2:T+1, :);
y = y(:, 2:T+1, :);

%% Calculate the Error
e_tfps = x-xhat_tfps;
e_ffbsi = x-xhat_ffbsi;

mse_tfps = [mean(e_tfps(1, :, :).^2 + e_tfps(2, :, :).^2, 3); ...
            mean(e_tfps(3, :, :).^2 + e_tfps(4, :, :).^2, 3)];
mse_ffbsi = [mean(e_ffbsi(1, :, :).^2 + e_ffbsi(2, :, :).^2, 3); ...
             mean(e_ffbsi(3, :, :).^2 + e_ffbsi(4, :, :).^2, 3)];

% save('large_sim.mat');

%% Export the Results
header = {'t', 'range', 'bearing', 'x1', 'x2', 'x1_tfps', 'x2_tfps', ...
    'x1_ffbsi', 'x2_ffbsi', 'mse_tfps_1', 'mse_tfps_2', 'mse_ffbsi_1', ...
    'mse_ffbsi_2'};
% iExample = randi(N, 1);
iExample = N;
data = [(1:T).', ...
        y(1, :, iExample).', ...
        y(2, :, iExample).', ...
        x(1, :, iExample).', ...
        x(2, :, iExample).', ...
        xhat_tfps(1, :, iExample).', ...
        xhat_tfps(2, :, iExample).', ...
        xhat_ffbsi(1, :, iExample).', ...
        xhat_ffbsi(2, :, iExample).', ...
        mean(e_tfps(1, :, :).^2 + e_tfps(2, :, :).^2, 3).', ...
        mean(e_tfps(3, :, :).^2 + e_tfps(4, :, :).^2, 3).', ...
        mean(e_ffbsi(1, :, :).^2 + e_ffbsi(2, :, :).^2, 3).', ...
        mean(e_ffbsi(3, :, :).^2 + e_ffbsi(4, :, :).^2, 3).'
];
filename = '/home/roland/Documents/Publications/2015 IEEE MSC Particle Smoother with Linear Process Dynamics/Article/Data/example_2.csv';
% csvexport(filename, header, data)

%% Illustrations
iExample = N;

figure(1); clf();
plot([x(1, :, iExample).', xhat_tfps(1, :, iExample).', xhat_ffbsi(1, :, iExample).'], ...
     [x(2, :, iExample).', xhat_tfps(2, :, iExample).', xhat_ffbsi(2, :, iExample).']);
figure(4); clf();
subplot(211);
plot(y(1, :, iExample));
title('Range');
subplot(212);
plot(y(2, :, iExample));
title('Bearing');
 
figure(2); clf();
for i = 1:4
    subplot(4, 1, i);
    plot([mean(e_tfps(i, :, :), 3).', ...
          mean(e_ffbsi(i, :, :), 3).']);
    legend('2F-PS', 'FFBSi');
    title('Mean Error');
end

figure(3); clf();
for i = 1:4
    subplot(4, 1, i);
    semilogy([var(e_tfps(i, :, :), [], 3).', ...
              var(e_ffbsi(i, :, :), [], 3).']);
    legend('2F-PS', 'FFBSi');
    title('Mean Squared Error');
end

figure(5); clf();
for i = 1:2
    subplot(2, 1, i);
    semilogy([mse_tfps(i, :).', mse_ffbsi(i, :).'])
    legend('2F-PS', 'FFBSi');
    title('Mean Squared Error (Position and Speed)');
end
