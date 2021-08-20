function [ms, Ps] = rtss(y, F, G, Q, R, m0, P0)
% # Rauch-Tung-Striebel smoother
% ## Usage
% * `[ms, Ps] = rtss(y, F, G, Q, R, m0, P0)`
% 
% ## Description
% Complete Rauch–Tung–Striebel smoother (including the forward filter) for
% linear, Gaussian state-space models of the form
%
%   x[n] = F x[n-1] + q[n]
%   y[n] = G x[n] + r[n]
%
% with q[n] ~ N(0, Q), r[n] = N(0, R), x[0] = N(m0, P0).
%
% ## Input
% * `y`: Measurement data.
% * `F`, `Q`: Dynamic model parameters.
% * `G`, `R`: Measurement model parameters.
% * `m0`, `P0`: Initial state model parameters.
%
% ## Output
% * `ms`: MMSE smoothed state estimate.
% * `Ps`: Smoothed covariance.
%
% ## Authors
% * 2013-present -- Roland Hostettler

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

    %% Parameter check
    narginchk(7, 7);
    dx = size(F, 1);
    [dy, N] = size(y);
    if dy > N
        y = y.';
    end
    [dy, N] = size(y);
    
    %% Memory Allocation
    % Filtering pass variables
    mp = zeros(dx, 1, N+1);   % Stores \hat{x}_{t+1|t}
    Pp = zeros(dx, dx, N+1);  % Stores P_{t+1|t}
    Kt = zeros(dx, dy, N+1);    % Stores the Kalman gain
    m = zeros(dx, 1, N+1); % Stores \hat{x}_{t|t}
    P = zeros(dx, dx, N+1);   % Stores P_{t|t}
    
    % Smoothing pass variables
    ms = zeros(dx, 1, N); % Stores \hat{x}_{t|T}
    Ps = zeros(dx, dx, N);   % Stores P_{t|T}
    Ct = zeros(dx, dx, N);    % Stores the smoothing gain
    
    %% Filtering (Forward) Pass
    % Initialization
    m(:, :, 1) = m0;
    P(:, :, 1) = P0;
    y = [zeros(dy, 1) y];
    
    % Iterations
    for n = 2:N+1
        % Prediction Update
        mp(:, :, n) = F*m(:, :, n-1);
        Pp(:, :, n) = F*P(:, :, n-1)*F' + Q;

        % Time Update 
        Kt(:, :, n) = Pp(:, :, n)*G'/(R+G*Pp(:, :, n)*G');
        m(:, :, n) = mp(:, :, n)+Kt(:, :, n)*(y(:, n)-G*mp(:, :, n));        
        P(:, :, n) = (eye(dx)-Kt(:, :, n)*G)*Pp(:, :, n);
    end
    
    % Remove the initial value
    m = m(:, :, 2:N+1);
    mp = mp(:, :, 2:N+1);
    Pp = Pp(:, :, 2:N+1);
    P = P(:, :, 2:N+1);

    %% Smoothing (Backward) Pass
    % Intitialization
    ms(:, :, N) = m(:, :, N);
    Ps(:, :, N) = P(:, :, N);
    
    % Iterations
    for n = N-1:-1:1
        Ct(:, :, n) = P(:, :, n)*F'/Pp(:, :, n+1);
        ms(:, :, n) = m(:, :, n) + Ct(:, :, n)*(ms(:, :, n+1) - mp(:, :, n+1));
        Ps(:, :, n) = P(:, :, n) + Ct(:, :, n)*(Ps(:, :, n+1)-Pp(:, :, n+1))*Ct(:, :, n)';
    end
end
