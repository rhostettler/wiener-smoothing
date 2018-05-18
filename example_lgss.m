% Two-Filter Particle Smoothing with Linear State Dynamics
%
% This is the first example in [1] where the performance of the two-filter
% particle smoother with linear state dynamics (Wiener system) is
% illustrated. The smoother is compared to the Rauch-Tung-Striebel smoother
% in order to illustrate the performance.
% 
% 2015-03-20 -- Roland Hostettler <roland.hostettler@ltu.se>

% Housekeeping
clear variables;

%% Parameters
T = 100; % time samples
N = 10; % MC sims

%% System Description
A = @(t) [1, 0.5; 0, 1];
Q = @(t) eye(2);
C = [1, 0];
mu0 = [5; 3];
Sigma0 = diag([10, 20]);
R = 1;

%% Models

% 2F-PS model
model.A = A;
model.Q = Q;
model.likelihood = @(y, x, t) normpdf(y.', (C*x).', sqrt(R).').';
model.mu0 = mu0;
model.Sigma0 = Sigma0;

% 2F-PS parameters
par.M = 500;
par.M_T = round(par.M/3);

%% System Simulation
% Preallocate
x = zeros([2, T+1, N]);
y = zeros([1, T+1, N]);
xhat_tfps = zeros([2, T, N]);
xhat_rtss = zeros([2, T, N]);

%eta(N);
for k = 1:N
    % Initialize
    x(:, 1, k) = model.mu0 + chol(model.Sigma0, 'lower')*randn([2, 1]);

    % Simulate the system
    for n = 2:T+1
        t = n-1;

        % Propagate the state
        x(:, n, k) = model.A(t)*x(:, n-1, k) + chol(model.Q(t), 'lower')*randn([2, 1]);

        % Measurement
        y(:, n, k) = C*x(:, n, k) + chol(R, 'lower')*randn([1, 1]);
    end
    
    %% Smoothing
    % 2F-PS
    [xhat_tfps(:, :, k), sys] = tfps(y(:, 2:T+1, k), model, par);
    
    % RTSS
    [tmp, Phat_ks] = ks(y(:, 2:T+1, k), A(0), eye(2), C, Q(0), R, mu0, Sigma0);
    xhat_rtss(:, :, k) = squeeze(tmp);
        
    %% Progress Info
    %eta();
end
%eta(0);

% Remove initial states from the states and measurements
x = x(:, 2:T+1, :);
y = y(:, 2:T+1, :);

%% Calculate the Error
e_tfps = x-xhat_tfps;
e_rtss = x-xhat_rtss;

%% Export the Results
header = {'t', 'e_tfps_1', 'e_tfps_2', 'e_rtss_1', 'e_rtss_2', ...
    'mse_tfps_1', 'mse_tfps_2', 'mse_rtss_1', 'mse_rtss_2'};
data = [(1:T).', ...
        mean(e_tfps(1, :, :), 3).', ...
        mean(e_tfps(2, :, :), 3).', ...
        mean(e_rtss(1, :, :), 3).', ...
        mean(e_rtss(2, :, :), 3).', ...
        var(e_tfps(1, :, :), [], 3).', ...
        var(e_tfps(2, :, :), [], 3).', ...
        var(e_rtss(1, :, :), [], 3).', ...
        var(e_rtss(2, :, :), [], 3).'
];
filename = '/home/roland/Documents/Publications/2015 IEEE MSC Particle Smoother with Linear Process Dynamics/Article/Data/example_1.csv';
% csvexport(filename, header, data)

%% Illustrations
figure(1); clf();
subplot(211);
plot([mean(e_tfps(1, :, :), 3).', ...
      mean(e_rtss(1, :, :), 3).']);
legend('2F-PS', 'RTSS');
title('Mean Error');
subplot(212);
plot([mean(e_tfps(2, :, :), 3).', ...
      mean(e_rtss(2, :, :), 3).']);

figure(2); clf();
subplot(211);
semilogy([var(e_tfps(1, :, :), [], 3).', ...
          var(e_rtss(1, :, :), [], 3).']);
legend('2F-PS', 'RTSS');
title('Mean Squared Error');
subplot(212);
semilogy([var(e_tfps(2, :, :), [], 3)', ...
          var(e_rtss(2, :, :), [], 3).']);
