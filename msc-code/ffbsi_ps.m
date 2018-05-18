function [xhat, sys] = ffbsi_ps(y, model, par)
% Forward Filtering-Backward Simulation Particle Smoother
%
% SYNOPSIS
%   [xhat, sys] = ffbsi_ps(y, model, par)
% 
% DESCRIPTION
%   <function/algorithm description incl. parameters>
%
% EXAMPLE
%   <possible example(s)>
% 
% SEE ALSO
%   <list functions that are similar/related here>
%
% REFERENCES
%   <references>
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@ltu.se>
%
% VERSION
%   $ ffbsi_ps.m $ Version 1.0 $ 2015-03-23 $
    
% TODO
%   * Parameter check, parameters
%   * Documentation
%   * Model check

    %% Parameter Check
    % TODO: Implement me
        
    %% Initialization
%     par.Mf = 1000;
%     par.Ms = par.Mf/2;
%     par.M_T = par.Mf/3;

    % Check the dimensions of the measurement vector
    [Ny, T] = size(y);
    if Ny > T
        y = y.';
    end
%     [~, T] = size(y);

    %% Forward Filter
    sys = forward_filter(y, model, par);
    
    %% Backward Simulation
    sys = backward_simulation(sys, model, par);

    %% 
    % Return the estimated state
    xhat = cat(2, sys(:).xhat);
end

function sys = forward_filter(y, model, par)
    % Get parameters
    Mf = par.Mf;
    Ms = par.Ms;
    M_T = par.M_T;
    
    % Check the dimensions of the measurement vector
    [~, T] = size(y);
    Nx = size(model.px0.rng(1), 1);

    %% Preallocation
    sys(T+2).xFilter = zeros([Nx, Mf]);
    sys(T+2).wFilter = zeros([1, Mf]);
    sys(T+2).xSmoother = zeros([Nx, Ms]);
    sys(T+2).wSmoother = zeros([1, Ms]);
    sys(T+2).xhat = zeros([Nx, 1]);
    sys(T+2).xhatFilter = zeros([Nx, 1]);
    sys(T+2).Phat = zeros([Nx, Nx]);

    %% Initialize
    sys(1).xFilter = model.px0.rng(Mf);
    sys(1).wFilter = 1/Mf*ones([1, Mf]);
    %sys(1).xhat = mean(sys(1).x, 2);
    sys(1).xhatFilter = mean(sys(1).xFilter, 2);
    %sys(1).Phat = model.Sigma0;

    % Forward recursion
    for n = 2:T+1
        t = n-1;
        
        % Propagate the particles
        x = model.px.rng(sys(n-1).xFilter, t);
        
        % Calculate the particle weight
        wFilter = sys(n-1).wFilter.*model.py.eval(y(:, t), x, t);
        wFilter = wFilter/sum(wFilter);

        % Estimate the filtered state and covariance
        sys(n).xhatFilter = x*wFilter';
        
        % Resample if the effective sample size falls below the threshold
        ess = 1/sum(wFilter.^2);
        if ess < M_T
            ir = resample(wFilter);
            x = x(:, ir);
            wFilter = 1/Mf*ones([1, Mf]);
        end

        % Store the samples and their weights
        sys(n).xFilter = x;
        sys(n).wFilter = wFilter;        
    end
    
    % Remove the leading state
    sys = sys(2:T+1);
end

%% 
function sys = backward_simulation(sys, model, par)
    %% Initialize
    Mf = par.Mf;
    Ms = par.Ms;
    
    T = length(sys);
    
    %% Initialize
    ir = resample(sys(T).wFilter);
    x = sys(T).xFilter(:, ir);
    b = randperm(Mf, Ms);
    sys(T).xSmoother = x(:, b);
    sys(T).wSmoother = 1/Ms*ones([1, Ms]);
    sys(T).xhat = mean(sys(T).xSmoother, 2);
    
    %% Backward Iteration
    % j -> trajectory to expand
    % i -> candidate particles
    for t = T-1:-1:1
        for j = 1:Ms
            % Compute the backward smoothing weights
%             wSmoother = zeros([1, Mf]);
%             for i = 1:Mf
%                 wSmoother(i) = sys(t).wFilter(i)*model.px.eval( ...
%                     sys(t+1).xSmoother(:, j), sys(t).xFilter(:, i), t);
%             end
            % This only works because the transition density is Gaussian!
            wSmoother = sys(t).wFilter.*model.px.eval( ...
                    sys(t+1).xSmoother(:, j), sys(t).xFilter, t);
            wSmoother = wSmoother/sum(wSmoother);
                        
            % Draw from the categorical distribution
            ir = resample(wSmoother);
            xp = sys(t).xFilter(:, ir);
            b = randi(Mf, 1);            
            
            % Extend the backwards trajectory
            sys(t).xSmoother(:, j) = xp(:, b);
            sys(t).wSmoother(:, j) = 1/Ms;
        end
        
        % Estimate the state
        sys(t).xhat = mean(sys(t).xSmoother, 2);
    end
end

%% 
function ir = resample(w)
    M = length(w);
    ir = zeros(1, M);
    c = cumsum(w);
    u = 1/M*rand(1)+(0:M-1)'/M;
    for j = 1:M
        ir(j) = find(c > u(j), 1, 'first');
    end
end
