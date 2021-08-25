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
        sys = initialize_sys(N, dx, J);
        sys(1).alpha = 1:J;
        sys(1).x = x;
        sys(1).w = exp(lw);
        sys(1).qstate = [];

        % Mean and covariance of p(x[n])
        sys(1).mx = model.px0.mean([], []);
        sys(1).Cxx = model.px0.cov([], []);
        sys(1).Cax = [];
    end
    
    % (Local) helper variables
    xhat = zeros(dx, N-1);
    my = zeros(dy, J);          % Likelihood mean, covariance, and cross-
    Cxy = zeros(dx, dy, J);     % covariance
    Cy = zeros(dy, dy, J);   
    mxp = zeros(dx, J);         % Proposal mean and covariance
    Cxp = zeros(dx, dx, J);
    
    lv = zeros(1, J);
        
    %% Filtering
    for n = 2:N
        %% Calculate the proposals
        % Calculate the approximations of the adjustment multipliers and
        % optimal proposal.
        
        % Prior mean and covariance. N.B.: The model is linear in the
        % dynamics, hence, we can calculate the prior mean for all
        % particles at once.
        mx = model.px.mean(x, theta(:, n));
        Cx = model.px.cov([], theta(:, n));
        for j = 1:J
            % Moments of the Gaussian joint approximation
            %                             _    _    _    _    _         _
            %                           (| x[n] |  |  mx  |  |  Cx, Cxy  |)
            % p(x[n], y[n] | x[n-1]) = N(|_y[n]_|; |_ my _|, |_ Cyx, Cy _|)
            % 
            [my(:, j), Cy(:, :, j), Cxy(:, :, j)] = calculate_moments_taylor(model, mx(:, j), Cx, theta(:, n));
            
            % Moments of the Gaussian approximation of the optimal proposal
            %
            % p(x[n] | y[n], x[n-1]) = N(x[n]; mxp, Cxp)
            %
            % with
            %
            % K = Cxy/Cy
            % mxp = mx + K*(y - my)
            % Cxp = Cx - K*Cy*K'
            K = Cxy(:, :, j)/Cy(:, :, j);
            mxp(:, j) = mx(:, j) + K*(y(:, n) - my(:, j));
            Cxp(:, :, j) = Cx - K*Cy(:, :, j)*K';
            Cxp(:, :, j) = (Cxp(:, :, j) + Cxp(:, :, j)')/2;

            % Resampling weights
            lv(j) = lw(j) + logmvnpdf(y(:, n), my(:, j), Cy(:, :, j));
        end
        
        %% Resample
        v = exp(lv-max(lv));
        v = v/sum(v);
        alpha = resample_stratified(v);

        %% Sample states and calculate weights
        for j = 1:J
            % Sample from the optimal proposal approximation
            LCxp = chol(Cxp(:, :, alpha(j))).';
            x(:, j) = mxp(:, alpha(j)) + LCxp*randn(dx, 1);
            
            % Moments of the importance weights' denominator
            %
            % p(y[n] | x[n], x[n-1]) = N(y[n]; myp, Cyp)
            L = Cxy(:, :, alpha(j))'/Cx;
            myp = my(:, alpha(j)) + L*(x(:, j) - mx(:, alpha(j)));
            Cyp = Cy(:, :, alpha(j)) - L*Cx*L';
            
            % Particle weights
            lw(j) = model.py.logpdf(y(:, n), x(:, j), theta(:, n)) ...
                - logmvnpdf(y(:, n), myp, Cyp);
        end
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        %% Calculate mean and covariance of p(x[n])        
        % p(x[n]) = N(x[n]; sys(n).mu, sys(n).Sigma)
        if return_sys
            F = model.px.jacobian([], theta(:, n));
            sys(n).mx = F*sys(n-1).mx;      % Mean E{x[n]}
            Cxx = F*sys(n-1).Cxx*F' + Cx;   % Covariance Cov{x[n]}
            sys(n).Cxx = (Cxx + Cxx')/2;
            sys(n).Cax = sys(n-1).Cxx*F';   % Cross-covariance Cov{x[n-1], x[n]}
        end

        %% Estimate & store results
        % MMSE state estimate
        xhat(:, n-1) = x*w';
                
        %% Store
        if return_sys
            sys(n).alpha = alpha;
        	sys(n).x = x;
            sys(n).w = w;
        end
    end
    
    %% Calculate joint filtering density
    if return_sys
        sys = calculate_particle_lineages(sys);
    end
end
