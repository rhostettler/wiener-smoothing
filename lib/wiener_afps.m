function varargout = wiener_afps(model, y, theta, J, sys)
% # Two filter particle smoother for Wiener state space models
% ## Usage
% * `xhat = wiener_afps(model, y)`
% * `xhat = wiener_afps(model, y, theta, J, sys)`
%
% ## Description
% Two (forward/backward) filter particle smoother for Wiener state space
% models using approximately fully adapted auxiliary particle filters for
% the forward and backward filters, with Gaussian approximations of the
% importance densities, according to [1].
%
% ## Input
% * `model`: Model struct.
% * `y`: Meausrement matrix.
% * `theta`: Optional model parameters.
% * `J`: Number of particles to use (optional, default: `100`).
% * `sys`: Optional particle system if another filter has been run. If
%   omitted, `wiener_apf()` is run.
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
    narginchk(2, 5);
    if nargin < 3 || isempty(theta)
        theta = NaN;
    end
    if nargin < 4 || isempty(J)
        J = 100;
    end
    
    % Expand 'theta'
    if size(theta, 2) == 1
        N = size(y, 2);
        theta = theta*ones(1, N);
    end
    
    %% Smoothing
    % If no filtered system is provided, run a Wiener APF first
    if nargin < 6 || isempty(sys)
        [~, sys] = wiener_apf(model, y, theta, J);
    end
    
    % Actual smoothing
    [varargout{1:nargout}] = smooth(model, y, theta, sys);
end

%% Smoothing function
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
    myb = zeros(dy, J);
    Cxyb = zeros(dx, dy, J);
    Cyb = zeros(dy, dy, J);
    mxpb = zeros(dx, J);
    Cxpb = zeros(dx, dx, J);
    lv = zeros(1, J);
    lnu = zeros(1, J);

    %% Initialize backward filter
    % Resample the particles such that they are approximately distributed
    % according to
    %
    % p(x[N] | y[N]) ~ p(y[N] | x[N]) p(x[N])
    xf = sys(N).x;
    wf = sys(N).w;
    for j = 1:J
        lv(j) = model.py.logpdf(y(:, N), xf(:, j), theta(:, N)) ...
            + logmvnpdf(xf(:, j).', sys(N).mx.', sys(N).Cxx);
    end
    v = exp(lv-max(lv));
    v = v/sum(v);
    beta = resample_stratified(v);
    xs = xf(:, beta);
    lwb = log(1/J)*ones(1, J);
    ws = wf(beta);
    xhat(:, N-1) = xs*ws';
    
    % Store the particle system   
    return_sys = nargout >= 2;
    if return_sys
        sys(N).xs = xs;
        sys(N).ws = ws;
        sys(N).wb = exp(lwb);
    end

    %% Backward filter and smoothing
    for n = N-1:-1:2
        %% Calculate the proposal
        % Get mean and covariance the joint density
        %                      _      _    _    _    _            _
        %                    (| x[n]   |  |  mx  |  | Cxx,  Cxxn   |)
        % p(x[n], x[n+1]) = N(|_x[n+1]_|; |_ mxn_|, |_Cxxn', Cxnxn_|)
        mx = sys(n).mx;
        Cxx = sys(n).Cxx;
        mxn = sys(n+1).mx;
        Cxnxn = sys(n+1).Cxx;
        Cxxn = sys(n+1).Cax;
                
        % Calculate the backward dynamic model
        %
        % p(x[n] | x[n+1]) = N(x[n]; mxb, Cxb)
        K = Cxxn/Cxnxn;
        mxb = mx + K*(xs - mxn);
        Cxb = Cxx - K*Cxnxn*K';
        for j = 1:J
            % Calculate the joint Gaussian approximation
            %                             _    _    _   _    _          _
            %                           (| x[n] |  | mxb |  | Cxb,  Cxyb |)
            % p(x[n], y[n] | x[n+1]) = N(|_y[n]_|; |_myb_|, |_Cyxb, Cyb _|)
            %
            % using Taylor series linearization around the prior mean mxb
            % and covariance Cxb.
            [myb(:, j), Cyb(:, :, j), Cxyb(:, :, j)] = calculate_moments_taylor(model, mxb(:, j), Cxb, theta(:, n));
            
            % Calculate the moments of the Gaussian approximation of the
            % optimal backward filter proposal
            %
            % p(x[n] | x[n+1], y[n]) = N(x[n]; mxpb, Cxpb)
            %
            % with
            %
            % L = Cxyb/Cyyb
            % mxpb = mxb + L*(y[n] - myb)
            % Cxpb = Cxb - L*Cyyb*L'
            L = Cxyb(:, :, j)/Cyb(:, :, j);
            mxpb(:, j) = mxb(:, j) + L*(y(:, n) - myb(:, j));
            Cxpb(:, :, j) = Cxb - L*Cyb(:, :, j)*L';
            Cxpb(:, :, j) = (Cxpb(:, :, j) + Cxpb(:, :, j)')/2;

            % Resampling weights
            lv(j) = lwb(j) + logmvnpdf(y(:, n).', myb(:, j).', Cyb(:, :, j)).';
        end
        
        %% Resampling
        v = exp(lv-max(lv));
        v = v/sum(v);
        lv = log(v);
        beta = resample_stratified(v);
        
        %% Sample states and calculate weights
        for j = 1:J           
            % Sample from the optimal proposal approximation
            LCxpb = chol(Cxpb(:, :, beta(j))).';
            xs(:, j) = mxpb(:, beta(j)) + LCxpb*randn(dx, 1);

            % Moments of the importance weights denominator
            %
            % p(y[n] | x[n], x[n-1]) = N(y[n]; mypb, Cypb)
            L = Cxyb(:, :, beta(j))'/Cxb;
            myp = myb(:, beta(j)) + L*(xs(:, j) - mxb(:, beta(j)));
            Cypb = Cyb(:, :, beta(j)) - L*Cxb*L';
            
            lwb(:, j) = model.py.logpdf(y(:, n), xs(:, j), theta(:, n)) ...
                - logmvnpdf(y(:, n).', myp.', Cypb);
        end
        wb = exp(lwb-max(lwb));
        wb = wb/sum(wb);
        lwb = log(wb);

        %% Smoothing
        x = sys(n-1).x;
        lw = log(sys(n-1).w);
        for j = 1:J
            tmp = lw + model.px.logpdf(xs(:, j), x, theta(:, n));
            lnu(j) = log(sum(exp(tmp)));
        end
        lws = lwb + lnu - logmvnpdf(xs.', mx.', Cxx).';
        ws = exp(lws-max(lws));
        ws = ws/sum(ws);
        xhat(:, n-1) = xs*ws.';

        %% Store
        if return_sys
            sys(n).beta = beta;
            sys(n).xs = xs;
            sys(n).wb = wb;
            sys(n).ws = ws;
        end
    end
end
