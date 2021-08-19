% Particle Smoothing Illustration for Wiener State-Space Systems
%
% Numerical illustration of the Wiener systems particle smoother with a
% Gaussian approximation of the optimal proposal from [1].
%
% References:
%   [1] R. Hostettler and T. B. Schon, "Particle Filtering and Two Filter
%       Smoothing for Wiener State-Space Systems", to appear.
% 
% 2017-today -- Roland Hostettler

% TODO:
% * Update code
% * Add disclaimer
% * Move this to main directory
% * Update reference above
% * Make a stripped down version of the whole thing for homepage (upon
%   acceptance)

% Housekeeping
clear variables;
addpath lib;
rng(211);
%addpath ~/Projects/smc-mcmc/src/

%% Parameters
export = false;      % Export the results (RMSE and runtime)

N = 100;            % No. of datapoints
L = 100;            % No. of MC simulations

Jf = 500;           % No. of particles in the filter, ...
Js = Jf/2;          % ...and in the smoother
K = 10;             % No. of trajectories to draw in PGAS

m0 = zeros(4, 1);   % Initial state
P0 = eye(4);        % Initial covariance
Q = 0.25^2*eye(4);  % Process noise covariance
R = 0.1;            % Measurement noise covariance

%% Smoother parameters
% FFBSi
par_ffbsi.rs = false;    % Use rejection sampling

% PGAS
par_cpfas.Kburnin = 0;  % No burn-in (convergence is quite good)
par_cpfas.Kmixing = 1;  % No extra mixing (mixes well)
par_cpfas.filter = @(y, t, model, q, M, par) wiener_cpfas(y, t, model, M, par);

%% Model
model = model_nonmonotonic(Q, R, m0, P0);

%% Preallocate
t = (1:N).';
u = zeros(1, N);
Nx = 4;
xs = zeros(Nx, N, L);
xhat_bpf = zeros(Nx, N, L);
xhat_apf = xhat_bpf;
xhat_afps = xhat_bpf;
xhat_ffbsi = xhat_bpf;
xhat_ksd = xhat_bpf;
xhat_cpfas = xhat_bpf;
xhat_cpfas2 = xhat_bpf;

t_bpf = zeros(1, L);
t_apf = t_bpf;
t_afps = t_bpf;
t_ffbsi = t_bpf;
t_ksd = t_bpf;
t_cpfas = t_bpf;
t_cpfas2 = t_bpf;

%% Simulate
%eta(L);
fh = pbar(L);
% parfor l = 1:L
for l = 1:L
    %% Simulate the system
    [x, y] = simulate_model(model, [], N);   % TODO: External function

    %% Estimate
    % Bootstrap PF
    ts = tic;
    xhat_bpf(:, :, l) = pf(model, y, [], Jf);
    t_bpf(l) = toc(ts);
    
    % APF w/ approximation of the optimal proposal
    ts = tic;
    xhat_apf(:, :, l) = wiener_apf(model, y, [], Jf);
    t_apf(l) = toc(ts);
    
    % 2FS
    ts = tic;
    xhat_afps(:, :, l) = wiener_afps(model, y, [], Jf);
    t_afps(l) = toc(ts);
    
if 0
    % FFBSi
    %model.px.fast = 0;
    ts = tic;
    [~, ~, sys_ffbsi] = wiener_gaapf(ys, t, model, Jf);
    [xhat_ffbsi(:, :, l), ~, sys_ffbsi] = ffbsi_ps(ys, t, model, [], Js, par_ffbsi, sys_ffbsi);
    t_ffbsi(l) = toc(ts);
    
    % KSD
    ts = tic;
    [~, ~, sys_ksd] = wiener_gaapf(ys, t, model, Jf);
    xhat_ksd(:, :, l) = ksd_ps(ys, t, model, [], Jf, [], sys_ksd);
    t_ksd(l) = toc(ts);

    % CPF-AS
    ts = tic;
    xhat_cpfas(:, :, l) = cpfas_ps(ys, t, model, [], Jf, K, par_cpfas);
    t_cpfas(l) = toc(ts);
    
    % CPF-AS (2)
    ts = tic;
    xhat_cpfas2(:, :, l) = cpfas_ps(ys, t, model, [], Jf, 2*K, par_cpfas);
    t_cpfas2(l) = toc(ts);
end
    
    %% Show progress
    %eta();
    pbar(l, fh);
end
%eta(0);
pbar(0, fh);

%% Errors
trms = @(e) squeeze(sqrt(mean(sum(e.^2, 1))));
rms = @(e) sqrt(sum(e.^2, 1));

e_bpf = xhat_bpf - xs;
rms_bpf = mean(rms(e_bpf), 3);
trmse_bpf = mean(trms(e_bpf));
tstd_bpf = std(trms(e_bpf));

e_apf = xhat_apf - xs;
rms_apf = mean(rms(e_apf), 3);
trmse_apf = mean(trms(e_apf));
tstd_apf = std(trms(e_bpf));

e_ffbsi = xhat_ffbsi - xs;
rms_ffbsi = mean(rms(e_ffbsi), 3);
trmse_ffbsi = mean(trms(e_ffbsi));
tstd_ffbsi = std(trms(e_ffbsi));

e_ksd = xhat_ksd - xs;
rms_ksd = mean(rms(e_ksd), 3);
trmse_ksd = mean(trms(e_ksd));
tstd_ksd = std(trms(e_ksd));

e_2fs = xhat_afps - xs;
rms_2fs = mean(rms(e_2fs), 3);
trmse_2fs = mean(trms(e_2fs));
tstd_2fs = std(trms(e_2fs));

e_cpfas = xhat_cpfas - xs;
rms_cpfas = mean(rms(e_cpfas), 3);
trmse_cpfas = mean(trms(e_cpfas));
tstd_cpfas = std(trms(e_cpfas));

e_cpfas2 = xhat_cpfas2 - xs;
rms_cpfas2 = mean(rms(e_cpfas2), 3);
trmse_cpfas2 = mean(trms(e_cpfas2));
tstd_cpfas2 = std(trms(e_cpfas2));

%% Performance
fprintf('\t\tRMSE\t\t\tTime\n')
fprintf('\t\t----\t\t\t----\n');
fprintf('Bootstrap PF:\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_bpf, tstd_bpf, mean(t_bpf), std(t_bpf));
fprintf('Auxiliary PF:\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_apf, tstd_apf, mean(t_apf), std(t_apf));
fprintf('2FS:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_2fs, tstd_2fs, mean(t_afps), std(t_afps));
fprintf('FFBSi:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ffbsi, tstd_ffbsi, mean(t_ffbsi), std(t_ffbsi));
fprintf('KSD:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ksd, tstd_ksd, mean(t_ksd), std(t_ksd));
fprintf('CPF-AS:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_cpfas, tstd_cpfas, mean(t_cpfas), std(t_cpfas));
fprintf('CPF-AS (2):\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_cpfas2, tstd_cpfas2, mean(t_cpfas2), std(t_cpfas2));

%% Export results
if export
    header = {'N', 'trmse', 'trmse_std', 't', 'tstd'};
    txtwrite('Results/bootstrap.txt', [Jf, trmse_bpf, tstd_bpf, mean(t_bpf), std(t_bpf)], header, [], true);
    txtwrite('Results/auxiliary.txt', [Jf, trmse_apf, tstd_apf, mean(t_apf), std(t_apf)], header, [], true);
    txtwrite('Results/2fs.txt', [Jf, trmse_2fs, tstd_2fs, mean(t_afps), std(t_afps)], header, [], true);
    txtwrite('Results/ffbsi.txt', [Jf, trmse_ffbsi, tstd_ffbsi, mean(t_ffbsi), std(t_ffbsi)], header, [], true);
    txtwrite('Results/ksd.txt', [Jf, trmse_ksd, tstd_ksd, mean(t_ksd), std(t_ksd)], header, [], true);
    txtwrite(sprintf('Results/cpfas_k%d.txt', K), [Jf, trmse_cpfas, tstd_cpfas, mean(t_cpfas), std(t_cpfas)], header, [], true);
    txtwrite(sprintf('Results/cpfas_k%d.txt', 2*K), [Jf, trmse_cpfas2, tstd_cpfas2, mean(t_cpfas2), std(t_cpfas2)], header, [], true);
end

%% Save workspace
filename = sprintf('Simulation Data/%s_Mf%d_Ms%d_K%d_L%d.mat', datestr(now, 'yyyymmdd_HHMM'), Jf, Js, K, L);
save(filename);
