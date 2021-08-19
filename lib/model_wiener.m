function model = model_wiener(F, Q, g, Gx, R, m0, P0)
% # Wiener state-space model
% ## Usage
% * `model = model_wiener(F, Q, g, R, m0, P0)`
%
% ## Description
% Defines the model structure for Wiener state-space models of the form
%
%   x[0] ~ N(m0, P0)
%   x[n] = F(t[n]) x[n-1] + q[n],
%   y[n] = g(x, t[n]) + r[n],
%
% where q[n] ~ N(0, Q[n]) and r[n] ~ N(0, R[n]).
%
% ## Input
% * `F`, `Q`: Matrices of the dynamic model.
% * `g`, `R`: Measurement model.
% * `m0`, `P0`: Mean and covariance of the initial state.
%
% ## Output
% * `model`: The model structure that contains the usual fields (the
%   probabilistic representation of the state-space model, i.e., px0, px, 
%   py).
%
% ## Authors
% 2018-today -- Roland Hostettler

    %% Defaults
    narginchk(7, 7);
    if ~isa(F, 'function_handle')
        F = @(theta) F;
    end
    if ~isa(Q, 'function_handle')
        Q = @(theta) Q;
    end
    if ~isa(R, 'function_handle')
        R = @(theta) R;
    end
    
    %% Model structure
    % Initial state distribution
    dx = size(m0, 1);
    L0 = chol(P0).';
    px0 = struct();
    px0.rand = @(M) m0*ones(1, M)+L0*randn(dx, M);
    px0.logpdf = @(x, theta) logmvnpdf(x.', m0.', P0.');
    px0.fast = true;
    px0.mean = @(x, theta) m0;
    px0.cov = @(x, theta) P0;
    
    % State transition densiy
    px = struct();
    px.rand = @(x, theta) F(theta)*x + chol(Q(theta)).'*randn(dx, size(x, 2));
    px.logpdf = @(xp, x, theta) logmvnpdf(xp.', (F(theta)*x).', Q(theta).').';
    px.fast = true;
    px.mean = @(x, theta) F(theta)*x;
    px.jacobian = @(x, theta) F(theta);
    px.cov = @(~, theta) Q(theta);
    
    
    % Likelihood
    % TODO: Doesn't work if g() actually depends on theta.
    dy = size(g(m0, []), 1);
    py = struct();
    py.rand = @(x, theta) g(x, theta) + chol(R(theta)).'*randn(dy, size(x, 2));
    py.logpdf = @(y, x, theta) logmvnpdf(y.', g(x, theta).', R(theta).').';
    py.mean = @(x, theta) g(x, theta);
    py.jacobian = @(x, theta) Gx(x, theta);
    py.cov = @(~, theta) R(theta);
    py.fast = false;
    
    % Complete model
    model = struct('px', px, 'py', py, 'px0', px0);
end
