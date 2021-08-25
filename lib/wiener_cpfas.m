function [x, sys] = wiener_cpfas(model, y, xtilde, theta, J)
% # Conditional particle filter with ancestor sampling for Wiener models
% ## Usage
% * `[x, sys] = wiener_cpfas(model, y, xtilde, theta, J)`
%
% ## Description
% Conditional particle filter with ancestor sampling for Wiener state-space
% models that uses a Taylor series based linearization to approximate the
% optimal proposal for ancestor indices, new particles, and ancestor
% sampling. This function is to be used as a state trajectory sampler in
% `gibbs_pmcmc()`.
%
% ## Input
% * `model`: Model struct.
% * `y`: Measurement data.
% * `xtilde`: Ancestor trajectory from previous iteration. If omitted, an
%   auxiliary particle filter for Wiener models (`wiener_apf()`) is run to
%   generate the initial trajecotry.
% * `theta`: Optional parameters.
% * `J`: Number of particles (default: `100`).
%
% ## Output
% * `x`: Newly sampled trajectory.
% * `sys`: Particle system.
%
% ## Author
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

    %% Defaults
    narginchk(2, 5);
    if nargin < 4 || isempty(theta)
        theta = NaN;
    end
    if nargin < 5 || isempty(J)
        J = 100;
    end

    %% Initialize seed trajectory
    % If no trajectory is given (e.g., for the first iteration), we draw an
    % initial trajectory from a bootstrap particle filter which helps to
    % speed up convergence.
    if nargin < 3 || isempty(xtilde) || all(all(xtilde == 0))
        % Run a pf and sample a trajectory according to the filter weights
        [~, tmp] = wiener_apf(model, y, theta, J);
        beta = resample_stratified(tmp(end).wf);
        j = beta(randi(J, 1));
        xf = cat(3, tmp.xf);
        [dx, ~, N] = size(xf);
        xtilde = reshape(xf(:, j, :), [dx, N]);
    end
    
    %% Initialize
    x = model.px0.rand(J);
    x(:, J) = xtilde(:, 1);
    lw = -log(J)*ones(1, J);
    
    % Prepend a NaN measurement (for x[0] where we don't have a 
    % measurement)
    [dy, N] = size(y);
    y = [NaN*ones(dy, 1), y];

    % Expand theta properly such that we have theta(:, n)
    [dtheta, Ntheta] = size(theta);
    if Ntheta == 1
        theta = theta*ones(1, N);
    end
    theta = [NaN*ones(dtheta, 1), theta];
    
    %% Preallocate
    dx = size(x, 1);
    N = N+1;
    my = zeros(dy, J);
    Cy = zeros(dy, dy, J);
    Cxy = zeros(dx, dy, J);
    xp = zeros(dx, J);
    lwp = zeros(1, J);
    mxp = zeros(dx, J);
    Cxp = zeros(dx, dx, J);
    lv = zeros(1, J);
    
    sys = initialize_sys(N, dx, J);
    sys(1).x = x;
    sys(1).w = exp(lw);
    sys(1).alpha = 1:J;
    sys(1).qstate = [];

    %% Iterate over the Data
    for n = 2:N
        %% Calculate the importance distribution's moments
        % Prior
        mx = model.px.mean(x, theta(:, n));
        Cx = model.px.cov([], theta(:, n));
        for j = 1:J
            % Proposal moments
            [my(:, j), Cy(:, :, j), Cxy(:, :, j)] = calculate_moments_taylor(model, mx(:, j), Cx, theta(:, n));
            K = Cxy(:, :, j)/Cy(:, :, j);
            mxp(:, j) = mx(:, j) + K*(y(:, n) - my(:, j));
            Cxp(:, :, j) = Cx - K*Cy(:, :, j)*K';
            
            % Resampling weights (non-normalized)
            lv(j) = lw(j) + logmvnpdf(y(:, n).', my(:, j).', Cy(:, :, j)).';
        end
        v = exp(lv-max(lv));
        v = v/sum(v);
        lv = log(v);

        %% Sample J-1 particles
        alpha = resample_stratified(v);
        for j = 1:J-1
            % Propagate the particles
            LCxp = chol(Cxp(:, :, alpha(j))).';
            xp(:, j) = mxp(:, alpha(j)) + LCxp*randn(dx, 1);
            L = Cxy(:, :, alpha(j))'/Cx;
            myp = my(:, alpha(j)) + L*(xp(:, j) - mx(:, alpha(j)));
            Cyp = Cy(:, :, alpha(j)) - L*Cx*L';
            
            % Updated particle weight
            lwp(:, j) = model.py.logpdf(y(:, n), xp(:, j), theta(:, n)) ...
                - logmvnpdf(y(:, n), myp, Cyp);            
        end

        %% Propagate the Jth particle
        xp(:, J) = xtilde(:, n);
        
        % Calculate ancestor weights
        lwtilde = lw + model.px.logpdf(xtilde(:, n)*ones(1, J), x, theta(:, n));
        wtilde = exp(lwtilde-max(lwtilde));
        wtilde = wtilde/sum(wtilde);
        tmp = resample_stratified(wtilde);
        alpha(J) = tmp(randi(J, 1));
        L = Cxy(:, :, alpha(J))'/Cx;
        myp = my(:, alpha(J)) + L*(xp(:, J) - mx(:, alpha(J)));
        Cyp = Cy(:, :, alpha(J)) - L*Cx*L';
        lwp(:, J) = model.py.logpdf(y(:, n), xp(:, J), theta(:, n)) ...
            - logmvnpdf(y(:, n), myp, Cyp);

        %% Update particles and weights
        x = xp;
        w = exp(lwp-max(lwp));
        w = w/sum(w);
        lw = log(w);
        
        %% Store
        sys(n).alpha = alpha;
        sys(n).x = x;
        sys(n).w = w;
        sys(n).qstate = [];
    end
        
    %% Sample trajectory
    beta = resample_stratified(w);
    j = beta(randi(J, 1));
    tmp = calculate_particle_lineages(sys, j);
    x = cat(2, tmp.xf);
end
