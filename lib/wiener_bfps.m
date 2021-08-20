function varargout = wiener_bfps(model, y, theta, J, sys)
% # Bootstrap filter based two filter smoother for Wiener state-space models
% ## Usage
% * `xhat = wiener_bfps(model, y)`
% * `[xhat, sys]` =wiener_bfps(model, y, theta, J, par, sys)
%
% ## Description
% Bootstrap particle filter based two filter smoother for Wiener state-
% space models according to [1].
%
% N.B.: The filters use resampling based on the effective sample size 
% (ESS). This is currently hard-coded and can not be changed.
%
% ## Input
% * `model`: Model struct.
% * `y`: Measurement data.
% * `theta`: Optional model parameters.
% * `J`: Number of particles (optional, default: `100`).
% * `par`: Additional algorithm parameters:
%     - `Jt`: Resampling threshold (default: `J/3`).
% * `sys`: Particle system if an external filter is used. Note that an
%   external filter must calculate the mean `sys(n).mx`, covariance
%   `sys(n).Cxx`, and cross-covariance `sys(n).Cax`.
%
% ## Output
% * `xhat`: MMSE state estimate.
% * `sys`: Particle system.
%
% ## References
% 1. R. Hostettler, "A two filter particle smoother for Wiener state-
%    space systems," in IEEE Conference on Control Applications (CCA),
%    Sydney, Australia, September 2015.
% 
% ## Authors
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
   
    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    
    %% Process data
    % If no filtered system is provided, run a bootstrap PF first
    if nargin < 5 || isempty(sys)
        sys = filter(model, y, theta, J);
    end
    [varargout{1:nargout}] = smooth(model, y, theta, sys);
end

%% Forward filter
function sys = filter(model, y, theta, J)
    %% Defaults
    narginchk(4, 4);
    
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
    
    sys = initialize_sys(N, dx, J);
    sys(1).x = x;
    sys(1).w = exp(lw);
    sys(1).alpha = 1:J;
    sys(1).mx = model.px0.mean([], []);
    sys(1).Cxx = model.px0.cov([], []);
    sys(1).Cax = [];
    
    %% Filtering
    for n = 2:N
        % Sample
        [xp, alpha, lq, qstate] = sample_bootstrap(model, y(:, n), x, lw, theta(:, n));
        lw = calculate_weights(model, y(:, n), xp, alpha, lq, x, lw, theta(:, n));
        w = exp(lw-max(lw));
        w = w/sum(w);
        lw = log(w);
        
        % Update state
        x = xp;
        
        % Calculate prior
        F = model.px.jacobian([], theta(:, n));
        Cx = model.px.cov([], theta(:, n));
        sys(n).mx = F*sys(n-1).mx;      % Mean E{x[n]}
        Cxx = F*sys(n-1).Cxx*F' + Cx;   % Covariance Cov{x[n]}
        sys(n).Cxx = (Cxx + Cxx')/2;
        sys(n).Cax = sys(n-1).Cxx*F';   % Cross-covariance Cov{x[n-1], x[n]}
        
        % Store
        sys(n).x = x;
        sys(n).w = w;
        sys(n).alpha = alpha;
        sys(n).qstate = qstate;
    end
end

%% Backward filter and smoothing
function [xhat, sys] = smooth(model, y, theta, sys)
    %% Defaults
    narginchk(4, 4);

    %% Preallocate
    [dy, N] = size(y);
    [dx, J] = size(sys(1).x);
    xhat = zeros(dx, N);
    y = [NaN*ones(dy, 1), y];
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];
    N = N+1;
    
    % Temporary variables
    lwb = zeros(1, J);
    lv = zeros(1, J);
    lnu = zeros(1, J);
    
    %% Initialization
    % Backward filter => Use particles from forward pass and reweigh them
    xs = sys(N).x;
    ws = sys(N).w;
    lpx = logmvnpdf(xs.', sys(N).mx.', sys(N).Cxx.').';
    for j = 1:J
        lwb(j) = model.py.logpdf(y(:, N), xs(:, j), theta(:, N)) ...
            + lpx(j) ...
            - model.px.logpdf(xs(:, j), sys(N-1).x(:, sys(N).alpha(j)), theta(:, N));
    end
%     lwb2 = log(ws) +  lpx - log(sys(N-1).w(sys(N).alpha)) ...
%         - model.px.logpdf(xs, sys(N-1).x(:, sys(N).alpha), theta(:, N));
    wb = exp(lwb-max(lwb));
    wb = wb/sum(wb);
    lwb = log(wb);
    
    % Store
    sys(N).xs = xs;
    sys(N).ws = ws;
    sys(N).wb = wb;

    % Smoothed state estimate
    xhat(:, N-1) = xs*ws';

    %% Backward recursion
    for n = N-1:-1:2
        %% Calculate the proposal distribution
        % Get mean and covariance the joint density
        %                      _      _    _    _    _            _
        %                    (| x[n]   |  |  mx  |  | Cxx,  Cxxn   |)
        % p(x[n], x[n+1]) = N(|_x[n+1]_|; |_ mxn_|, |_Cxxn', Cxnxn_|)
        mx = sys(n).mx;
        Cxx = sys(n).Cxx;
        mxn = sys(n+1).mx;
        Cxnxn = sys(n+1).Cxx;
        Cxxn = sys(n+1).Cax;
        
        % Calculate the proposal distribution (backward dynamic model)
        %
        %  p(x[n] | x[n+1]) = N(x[n]; mxp, Cxp)
        %
        % with
        %
        % K = Cxxn/Cxnxn
        % mxp = mx + K*(xs - mxp)
        % Cxp = Cxx - K*Cxnxn*K'
        K = Cxxn/Cxnxn;
        mxp = mx + K*(xs - mxn);
        Cxp = Cxx - K*Cxnxn*K';
        Cxp = (Cxp + Cxp')/2;

        %% Sample
        [beta, ~, rstate] = resample_ess(lwb);
        if rstate.r
            lwb = -log(J)*ones(1, J);
        end
        LCxp = chol(Cxp).';
        xp = mxp(:, beta) + LCxp*randn(dx, J);
        
        % Calculate backward filter weights
        if model.py.fast
            lv = model.py.logpdf(y(:, n), xp, theta(:, n));
        else
            for j = 1:J
                lv(j) = model.py.logpdf(y(:, n), xp(:, j), theta(:, n));
            end
        end
        lwb = lwb + lv;
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);
        lwb = log(wb);
        xs = xp;

        %% Smoothing
        for j = 1:J
            tmp = log(sys(n-1).w) + model.px.logpdf(xs(:, j), sys(n-1).x, theta(:, n));
            lnu(j) = log(sum(exp(tmp)));
        end
        lws = lwb + lnu - logmvnpdf(xs.', sys(n).mx.', sys(n).Cxx).';
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
                
        %% Point estimate
        xhat(:, n-1) = xs*ws';
        
        %% Store
        sys(n).xs = xs;
        sys(n).wb = wb;
        sys(n).ws = ws;
        sys(n).beta = beta;
        sys(n).rs = rstate;
    end
end
