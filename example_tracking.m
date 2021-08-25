% Two filter bootstrap particle smoothing for Wiener state-space models
%
% This is the second example in [1] where the two filter particle smoother
% is applied to a bearings-only tracking problem.
%
% [1] R. Hostettler, "A two filter particle smoother for Wiener state-space
%     systems," in IEEE Conference on Control Applications (CCA), Sydney, 
%     Australia, September 2015.
% 
% 2015-present -- Roland Hostettler

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
rng(1811);

%% Parameters
N = 40;     % No. of time samples
K = 20;     % No. of Monte Carlo simulations
J = 500;    % No. of particles
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
model = model_wiener(F, Q, g, [], R, m0, P0);
model.py.fast = true;

%% Simulation
% Pre-allocate
x = zeros([4, N, K]);
y = zeros([2, N, K]);
xhat_bfps = zeros([4, N, K]);
xhat_ffbsi = zeros([4, N, K]);
t_bfps = zeros(1, K);
t_ffbsi = zeros(1, K);

% Monte Carlo simulation
fh = pbar(K);
for k = 1:K
    % Generate data
    [x(:, :, k), y(:, :, k)] = simulate_model(model, [], N);
    
    % Two filter smoother
    tstart = tic();
    xhat_bfps(:, :, k) = wiener_bfps(model, y(:, :, k), [], J);
    t_bfps(k) = toc(tstart);

    % FFBSi
    tstart = tic();
    xhat_ffbsi(:, :, k) = ps(model, y(:, :, k), [], 2*J, J);
    t_ffbsi(k) = toc(tstart);
    
    % Progress update
    pbar(k, fh);
end
pbar(0, fh);

%% Error
trms = @(e) squeeze(sqrt(mean(sum(e.^2, 1))));
rms = @(e) sqrt(sum(e.^2, 1));

e_bfps = x-xhat_bfps;
e_ffbsi = x-xhat_ffbsi;

rms_bfps = mean(rms(e_bfps), 3);
trmse_bfps = mean(trms(e_bfps));
tstd_bfps = std(trms(e_bfps));

rms_ffbsi = mean(rms(e_ffbsi), 3);
trmse_ffbsi = mean(trms(e_ffbsi));
tstd_ffbsi = std(trms(e_ffbsi));

%% Performance
fprintf('\tRMSE\t\t\tTime\n')
fprintf('\t----\t\t\t----\n');
fprintf('BFPS:\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_bfps, tstd_bfps, mean(t_bfps), std(t_bfps));
fprintf('FFBSi:\t%.2e (%.2e)\t%.3f (%.3f)\n', trmse_ffbsi, tstd_ffbsi, mean(t_ffbsi), std(t_ffbsi));
         
%% Illustrations
mse_bfps = [mean(e_bfps(1, :, :).^2 + e_bfps(2, :, :).^2, 3); ...
            mean(e_bfps(3, :, :).^2 + e_bfps(4, :, :).^2, 3)];
mse_ffbsi = [mean(e_ffbsi(1, :, :).^2 + e_ffbsi(2, :, :).^2, 3); ...
             mean(e_ffbsi(3, :, :).^2 + e_ffbsi(4, :, :).^2, 3)];

iExample = K;

figure(1); clf();
plot( ...
    [x(1, :, iExample).', xhat_bfps(1, :, iExample).', xhat_ffbsi(1, :, iExample).'], ...
    [x(2, :, iExample).', xhat_bfps(2, :, iExample).', xhat_ffbsi(2, :, iExample).'] ...
);
legend('Trajectory', 'BFPS', 'FFBSi');

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
    plot([mean(e_bfps(i, :, :), 3).', mean(e_ffbsi(i, :, :), 3).']);
    legend('BFPS', 'FFBSi');
    title('Mean Error');
end

figure(4); clf();
for i = 1:4
    subplot(4, 1, i);
    plot([var(e_bfps(i, :, :), [], 3).', var(e_ffbsi(i, :, :), [], 3).']);
    legend('BFPS', 'FFBSi');
    title('Mean Squared Error');
end

figure(5); clf();
for i = 1:2
    subplot(2, 1, i);
    plot([mse_bfps(i, :).', mse_ffbsi(i, :).'])
    legend('BFPS', 'FFBSi');
    title('Mean Squared Error (Position and Speed)');
end
