% Two-Filter Particle Smoothing with Linear State Dynamics
%
% 
% 
% 2015-03-20 -- Roland Hostettler <roland.hostettler@ltu.se>

% Housekeeping
clear variables;

%% Parameters
T = 10; % time samples
N = 10; % MC sims

%% System Description
A = @(t) [1, 0.5; 0, 1];
Q = @(t) eye(2);
C = [1, 0];
mu0 = [5; 3];
Sigma0 = diag([10, 20]);
R = 0.25;

model.A = A;
model.Q = Q;
model.likelihood = @(y, x, t) normpdf(y.', (C*x).', sqrt(R).').';
model.mu0 = mu0;
model.Sigma0 = Sigma0;

% FFBSi Model
model_ffbsi = struct();
model_ffbsi.px0.rng = @(M) mu0*ones([1, M]) + chol(Sigma0, 'lower')*randn([2, M]);
model_ffbsi.px.rng = @(x, t) A(t)*x + chol(Q(t), 'lower')*randn([size(Q(t), 1), size(x, 2)]);
model_ffbsi.px.eval = @(xp, x, t) mvnpdf(xp.', (A(t)*x).', Q(t).').';
model_ffbsi.py.eval = @(y, x, t) normpdf(y.', (C*x).', sqrt(R).').';
model_ffbsi.px.rho = model_ffbsi.px.eval(zeros([2, 1]), zeros([2, 1]), 0);

%% System Simulation
% Preallocate
x = zeros([2, T+1]);
%y = zeros([1, T+1]);
y0 = zeros([1, T+1]);
xhat = zeros([2, T, N]);
xhat_ks = zeros([2, T, N]);
xhatFilter = zeros([2, T, N]);
xhat_ffbsi = zeros([2, T, N]);

% Initialize
x(:, 1) = model.mu0 + chol(model.Sigma0, 'lower')*randn([2, 1]);

for n = 2:T+1
    t = n-1;

    % Propagate the state
    x(:, n) = model.A(t)*x(:, n-1) + chol(model.Q(t), 'lower')*randn([2, 1]);

    % Measurement
    y0(:, n) = C*x(:, n);
end

y0 = y0(:, 2:T+1);



M = 1000;
M_T = round(M/3);
par_tfps.M = M;
par_tfps.M_T = M_T;

par_ffbsi.Mf = M;
par_ffbsi.M_T = M_T;
par_ffbsi.Ms = M/2;

eta(N);
for k = 1:N
    % Simulate the system
    y = y0+sqrt(R)*randn(1, T);
    
    %% Smoothing
    par = [];
    [xhat(:, :, k), sys] = tfps(y, model, par_tfps);
    xhatFilter(:, :, k) = cat(2, sys(:).xhatFilter);
    
    [tmp, Phat_ks] = ks(y, A(0), eye(2), C, Q(0), R, mu0, Sigma0);
    xhat_ks(:, :, k) = squeeze(tmp(:, :, :));
    
%     xhat_ffbsi(:, :, k) = ffbsi_ps(y, model_ffbsi, par_ffbsi);
%     xhat_ffbsi(:, :, k) = rsffbsi_ps(y, model_ffbsi);
    
    %%
    eta();
end
    
x = x(:, 2:T+1, :);
% y = y(:, 2:T+1, :);

%% Calculate the Error
X = repmat(x, [1, 1, N]);
e_tfps = X-xhat;
e_pf = X-xhatFilter;
e_ks = X-xhat_ks;
e_ffbsi = X-xhat_ffbsi;

%% Illustrations
figure(2); clf();
subplot(211);
plot([mean(e_tfps(1, :, :), 3).', ...
      mean(e_pf(1, :, :), 3).', ...
      mean(e_ks(1, :, :), 3).', ...
      mean(e_ffbsi(1, :, :), 3).']);
legend('Smoothed', 'Filtered', 'RTS', 'FFBSi');
title('Error');
subplot(212);
plot([mean(e_tfps(2, :, :), 3).', ...
      mean(e_pf(2, :, :), 3).', ...
      mean(e_ks(2, :, :), 3).', ...
      mean(e_ffbsi(2, :, :), 3).']);

figure(3); clf();
subplot(211);
semilogy([var(e_tfps(1, :, :), [], 3).', ...
          var(e_pf(1, :, :), [], 3).', ...
          var(e_ks(1, :, :), [], 3).', ...
%           var(e_ffbsi(1, :, :), [], 3).' ...
]);
legend('Smoothed', 'Filtered', 'RTS', 'FFBSi');
title('Variance');
subplot(212);
semilogy([var(e_tfps(2, :, :), [], 3).', ...
          var(e_pf(2, :, :), [], 3).', ...
          var(e_ks(2, :, :), [], 3).', ...
%           var(e_ffbsi(2, :, :), [], 3).' ...
]);

%% 
xBackward = cat(3, sys(:).xBackward);
xForward = cat(3, sys(:).xForward);
figure(4); clf();
subplot(211);
plot(squeeze(xBackward(1, :, :)).', 'ob'); hold on;
plot(squeeze(xForward(1, :, :)).', '.r');

subplot(212);
plot(squeeze(xBackward(2, :, :)).', 'ob'); hold on;
plot(squeeze(xForward(2, :, :)).', '.r');


%% 
PhatFilter = cat(3, sys(:).PhatFilter);
Phat = cat(3, sys(:).Phat);

figure(5); clf();
subplot(211);
semilogy([squeeze(Phat_ks(1, 1, :)), squeeze(Phat(1, 1, :)), squeeze(PhatFilter(1, 1, :))]);
subplot(212);
semilogy([squeeze(Phat_ks(2, 2, :)), squeeze(Phat(2, 2, :)), squeeze(PhatFilter(2, 2, :))]);
