% Particle Smoothing Illustration for Wiener State-Space Systems
%
% Numerical illustration for the Wiener systems particle smoother with a
% Gaussian approximation of the optimal proposal from [1].
%
% References:
%   [1] R. Hostettler and T. Sch?n, "Particle Filtering and Two Filter
%       Smoothing for Wiener State-Space Systems", to appear.
% 
% wiener_example.m -- 2017-01-20
% Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;

%% Parameters
% No. of datapoints
N = 100;

% No of particles
Mf = 100;
Ms = 100;
K = Mf;

% No. of MC simulations
L = 1;

% Process dynamics covariance
var_q = 1;

% Measurement noise covariance
R = 0.5^2;

% Initial state
m0 = zeros(5, 1);
P0 = eye(5);

%% Model
model = regression_model(var_q, R, m0, P0);

%% 
t = (1:N).';
Nx = length(m0);
xs = zeros(Nx, N, L);
xhat_bpf = xs;
xhat_ffbsi = xs;

xhat_2fws = xhat_bpf;
xhat_ffbsi2 = xhat_bpf;
xhat_cpfas = xhat_bpf;

t_bpf = zeros(1, L);
t_ffbsi = t_bpf;

t_2fws = t_bpf;

t_ffbsi2 = t_bpf;
t_cpfas = t_bpf;

%% Simulate
%parfor l = 1:L
%eta(L);
for l = 1:L    
    %% Simulate the System
    x = model.px0.rand(1);
    ys = zeros(1, N);

    for n = 1:N
        % Propagate
        r = sqrt(R)*randn(1);
        x = model.px.rand(x, t(n));
        y = model.g(x, r, t(n));

        % Store
        xs(:, n, l) = x;
        ys(:, n) = y;   
    end
    y = ys;

    %% Estimate
    % Bootstrap PF
    ts = tic;
    xhat_bpf(:, :, l) = bootstrap_pf(ys, t, model, Mf);
    t_bpf(l) = toc(ts);

    % 
    ts = tic;
    %[~, ~, sys_ffbsi] = bootstrap_pf(ys, t, model, Ms);
    xhat_2fws(:, :, l) = wiener_2fs2(ys, t, model, Ms);
    t_ffbsi(l) = toc(ts);
    t_2fws(l) = toc(ts);
    
if 0
    % 2FS
    ts = tic;
    xhat_2fws(:, :, l) = smoother.smooth(ys, t, u);
    t_2fws(l) = toc(ts);

    
    % FFBSi*
    ts = tic;
    xhat_ffbsi2(:, :, l) = ffbsi2.smooth(ys, t, u);
    t_ffbsi2(l) = toc(ts);

    % CPF-AS
    ts = tic;
    xhat_cpfas(:, :, l) = cpfas.smooth(ys, t, u);
    t_cpfas(l) = toc(ts);
end
    
    %% Show Progress
%    eta();
end
%eta(0);

%% Errors
trms = @(e) squeeze(mean(sqrt(sum(e.^2, 1))));
rms = @(e) sqrt(sum(e.^2, 1));

e_bpf = xhat_bpf - xs;
rms_bpf = mean(rms(e_bpf), 3);
trmse_bpf = mean(trms(e_bpf));
tstd_bpf = std(trms(e_bpf));

e_ffbsi = xhat_ffbsi - xs;
rms_ffbsi = mean(rms(e_ffbsi), 3);
trmse_ffbsi = mean(trms(e_ffbsi));
tstd_ffbsi = std(trms(e_ffbsi));

e_2fs = xhat_2fws - xs;
rms_2fs = mean(rms(e_2fs), 3);
trmse_2fs = mean(trms(e_2fs));
tstd_2fs = std(trms(e_2fs));

% e_ffbsi2 = xhat_ffbsi2 - xs;
% rms_ffbsi2 = mean(rms(e_ffbsi2), 3);
% trmse_ffbsi2 = mean(trms(e_ffbsi2));
% tstd_ffbsi2 = std(trms(e_ffbsi2));

% e_cpfas = xhat_cpfas - xs;
% rms_cpfas = mean(rms(e_cpfas), 3);
% trmse_cpfas = mean(trms(e_cpfas));
% tstd_cpfas = std(trms(e_cpfas));

%% Performance
fprintf('\t\tRMSE\t\t\tTime\n')
fprintf('\t\t----\t\t\t----\n');
fprintf('Bootstrap PF:\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_bpf, tstd_bpf, mean(t_bpf), std(t_bpf));
fprintf('2FS:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_2fs, tstd_2fs, mean(t_2fws), std(t_2fws));
%fprintf('FFBSi:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ffbsi, tstd_ffbsi, mean(t_ffbsi), std(t_ffbsi));
%fprintf('KSD:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ksd, tstd_ksd, mean(t_ksd), std(t_ksd));
