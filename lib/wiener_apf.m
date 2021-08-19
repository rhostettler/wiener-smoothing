function [xhat, sys] = wiener_apf(model, y, theta, J)
% # Auxiliary particle filter for Wiener state space models
% ## Usage
% * `xhat = wiener_apf(model, y, theta)`
% * `[xhat, sys] = wiener_apf(model, y, theta, J)
%
% ## Description
% (Approximately) fully adapted auxiliary particle filter for Wiener state 
% space models with Gaussian approximation of the optimal adjustment
% multipliers and proposal distribution according to [1].
%
% ## Input
% * `model`: Model struct.
% * `y`: Measurement matrix.
% * `theta`: Optional parameters.
% * `J`: Number of particles (optional, default: `100`).
%
% ## Output
% * `xhat`: MMSE state estimate.
% * `sys`: Particle system.
%
% ## References
% * R. Hostettler and T. B. Schön, “Auxiliary-particle-filter-based two-
%   filter smoothing for Wiener state-space models,” in 21th International 
%   Conference on Information Fusion (FUSION), Cambridge, UK, July 2018
%
% ## Authors
% * 2017-present -- Roland Hostettler

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

% TODO:
% * How to handle dependencies?
% * Clean up. We can calculate almost everything in one for loop, the
%   second should only be needed for sampling and weight calculation.

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end

    %% Initialization
    x = model.px0.rand(J);
    lw = log(1/J)*ones(1, J);
    
    % Since we also store and return the initial state (in 'sys'), a dummy
    % (NaN) measurement is prepended to the measurement matrix so that we 
    % can use consistent indexing in the processing loop.
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];
    
    % Expand 'theta' to the appropriate size, such that we can use
    % 'theta(:, n)' as an argument to the different functions (if not 
    % already expanded).
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];

    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    
    % Particle system
    return_sys = (nargout >= 2);
    if return_sys
        sys = initialize_sys(N, dx, J);     % TODO: function to copy here
        sys(1).alpha = 1:J;
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).qstate = [];

        % Mean and covariance of p(x[n])
        sys(1).mu = model.px0.mean([], []);
        sys(1).Sigma = model.px0.cov([], []);
    end
    
    % (Local) helper variables
    xhat = zeros(dx, N-1);
    mu_xp = zeros(dx, J);
    mu_yp = zeros(dy, J);
    B = zeros(dx, dy, J);
    S = zeros(dy, dy, J);
    mu_xn = zeros(dx, J);
    C_xn = zeros(dx, dx, J);
    lv = zeros(1, J);
    xn = zeros(dx, J);
        
    %% Filtering
    for n = 2:N
        %% Calculate joint approximations and resampling weights
        for j = 1:J
            mu_xp(:, j) = model.px.mean(x(:, j), theta(:, n));
            Q = model.px.cov([], theta(:, n));
            [mu_yp(:, j), S(:, :, j), B(:, :, j)] = calculate_moments_taylor(model, x(:, j), theta(:, n));           
            K = B(:, :, j)/S(:, :, j);
            
            mu_xn(:, j) = mu_xp(:, j) + K*(y(:, n) - mu_yp(:, j));
            C_xn(:, :, j) = Q - K*S(:, :, j)*K';
            C_xn(:, :, j) = (C_xn(:, :, j) + C_xn(:, :, j)')/2;

            lv(j) = lw(j) + logmvnpdf(y(:, n), mu_yp(:, j), S(:, :, j));
        end
        
        %% Resample
        v = exp(lv-max(lv));
        v = v/sum(v);
        alpha = catrnd(v, [1, J]);       % TODO: Get local copy of the function

        %% Draw new samples and calculate weigths
        for j = 1:J
            LC_xn = chol(C_xn(:, :, alpha(j))).';
            xn(:, j) = mu_xn(:, alpha(j)) + LC_xn*randn(dx, 1);
            
            L = B(:, :, alpha(j))'/Q;
            mu_yn = mu_yp(:, alpha(j)) + L*(xn(:, j) - mu_xp(:, alpha(j)));
            Sigma_yn = S(:, :, alpha(j)) - L*Q*L';
            
            lw(j) = model.py.logpdf(y(:, n), xn(:, j), theta(:, n)) ...
                - logmvnpdf(y(:, n), mu_yn, Sigma_yn);
        end
        x = xn;
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        %% Calculate mean and covariance of p(x[n])
        if return_sys
            F = model.px.jacobian([], theta);
            sys(n).mu = F*sys(n-1).mu;
            Sigma = F*sys(n-1).Sigma*F' + Q;
            sys(n).Sigma = (Sigma + Sigma')/2;
        end

        %% Estimate & store results
        % MMSE state estimate
        xhat(:, n-1) = xn*w';
                
        %% Store
        if return_sys
            sys(n).alpha = alpha;
        	sys(n).x = x;
            sys(n).w = w;
        end
    end
end
