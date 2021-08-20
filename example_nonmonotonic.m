% Example of two filter particle smoothing for Wiener state space models
% from [1].
%
% References
% * R. Hostettler and T. B. Schön, “Auxiliary-particle-filter-based two-
%   filter smoothing for Wiener state-space models,” in 21th International 
%   Conference on Information Fusion (FUSION), Cambridge, UK, July 2018
%
% 2017-present -- Roland Hostettler

%{
% This file is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by thee 
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
% 
% This file is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
% FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
% more details.
% 
% You should have received a copy of the GNU General Public License along 
% with this file. If not, see <http://www.gnu.org/licenses/>.
%}

% Housekeeping
clear variables;
addpath lib;
% addpath external;
rng(211);

%% Parameters
N = 100;            % No. of datapoints (100)
L = 10;            % No. of MC simulations (100)

Jf = 500;           % No. of particles in the filter, ...
Js = Jf/2;          % ...and in the FFBSi smoother
K = 10;             % No. of trajectories to draw in PGAS

m0 = zeros(4, 1);   % Initial state
P0 = eye(4);        % Initial covariance
Q = 0.25^2*eye(4);  % Process noise covariance
R = 0.1;            % Measurement noise covariance

%% Smoother parameters
par_ksd = struct('smooth', @smooth_ksd);

% TODO: Update these when re-enabling
% PGAS
par_cpfas.Kburnin = 0;  % No burn-in (convergence is quite good)
par_cpfas.Kmixing = 1;  % No extra mixing (mixes well)
par_cpfas.filter = @(y, t, model, q, M, par) wiener_cpfas(y, t, model, M, par);

%% Model
model = model_nonmonotonic(Q, R, m0, P0);

%% Preallocate
dx = 4;
xs = zeros(dx, N, L);
xhat_bpf = zeros(dx, N, L);
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
fh = pbar(L);
for l = 1:L
    %% Simulate the system
    [xs(:, :, l), y] = simulate_model(model, [], N);

    %% Estimate
    % Bootstrap PF
    ts = tic;
    xhat_bpf(:, :, l) = pf(model, y, [], Jf);
    t_bpf(l) = toc(ts);
    
    % Wiener APF with Gaussian approximation
    ts = tic;
    xhat_apf(:, :, l) = wiener_apf(model, y, [], Jf);
    t_apf(l) = toc(ts);
    
    % Wiener two filter smoother with APFs and Gaussian approximations
    ts = tic;
    xhat_afps(:, :, l) = wiener_afps(model, y, [], Jf);
    t_afps(l) = toc(ts);
    
    % FFBSi
    ts = tic;
    [~, sys_ffbsi] = wiener_apf(model, y, [], Jf);
    xhat_ffbsi(:, :, l) = ps(model, y, [], Jf, Js, [], sys_ffbsi);
    t_ffbsi(l) = toc(ts);
    
    % KSD
    ts = tic;
    [~, sys_ksd] = wiener_apf(model, y, [], Jf);
    xhat_ksd(:, :, l) = ps(model, y, [], Jf, Js, par_ksd, sys_ksd);
    t_ksd(l) = toc(ts);

if 0
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
    pbar(l, fh);
end
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

e_afps = xhat_afps - xs;
rms_afps = mean(rms(e_afps), 3);
trmse_afps = mean(trms(e_afps));
tstd_afps = std(trms(e_afps));

e_ffbsi = xhat_ffbsi - xs;
rms_ffbsi = mean(rms(e_ffbsi), 3);
trmse_ffbsi = mean(trms(e_ffbsi));
tstd_ffbsi = std(trms(e_ffbsi));

e_ksd = xhat_ksd - xs;
rms_ksd = mean(rms(e_ksd), 3);
trmse_ksd = mean(trms(e_ksd));
tstd_ksd = std(trms(e_ksd));

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
fprintf('AFPS:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_afps, tstd_afps, mean(t_afps), std(t_afps));
fprintf('FFBSi:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ffbsi, tstd_ffbsi, mean(t_ffbsi), std(t_ffbsi));
fprintf('KSD:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ksd, tstd_ksd, mean(t_ksd), std(t_ksd));
fprintf('CPF-AS:\t\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_cpfas, tstd_cpfas, mean(t_cpfas), std(t_cpfas));
fprintf('CPF-AS (2):\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_cpfas2, tstd_cpfas2, mean(t_cpfas2), std(t_cpfas2));

% Save workspace
% filename = sprintf('Simulation Data/%s_Mf%d_Ms%d_K%d_L%d.mat', datestr(now, 'yyyymmdd_HHMM'), Jf, Js, K, L);
% save(filename);
