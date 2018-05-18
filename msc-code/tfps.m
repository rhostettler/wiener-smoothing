function [xhat, sys] = tfps(y, model, par)
% Two-Filter Particle Smoohter for Systems with Linear State Dynamics

    % Check if all parameters are given
    if nargin < 3
        error('Not enough input parameters.')
    end
    
%     par.M = 1000;
%     par.M_T = M/3;

    % Check the dimensions of the measurement vector
    [Ny, T] = size(y);
    if Ny > T
        y = y.';
    end
    
    %% Forward Filter
    sys = forward_filter(y, model, par);
        
    %% Backward Filter
    sys = backward_filter(sys, y, model, par);
    
    %% Post-Processing
    xhat = cat(2, sys(:).xhat);
end

%% Forward Filter
function sys = forward_filter(y, model, par)

    %% Preparations
    % Get the filter parameters
    M = par.M;
    M_T = par.M_T;
    jitter = false;

    % Get the dimensions of the state and measurement data
    Nx = size(model.A(0), 1);
    T = size(y, 2);

    % Preallocation
    sys(T+2).xForward = zeros([Nx, M]);
    sys(T+2).wForward = zeros([1, M]);
    sys(T+2).xhatFilter = zeros([Nx, 1]);
    sys(T+2).PhatFilter = zeros([Nx, Nx]);
    sys(T+2).Sigma = zeros([Nx, Nx]);
    sys(T+2).mu = zeros([Nx, 1]);

    %% Initialize
    sys(1).xForward = model.mu0*ones([1, M]) ...
        + chol(model.Sigma0, 'lower')*randn([Nx, M]);
    sys(1).wForward = 1/M*ones([1, M]);
    sys(1).Sigma = model.Sigma0;
    sys(1).mu = model.mu0;
    sys(1).xhat = model.mu0;
    sys(1).xhatFilter = model.mu0;
    sys(1).PhatFilter = model.Sigma0;

    %% Recursion
    for n = 2:T+1
        t = n-1;
        
        % Propagate the particles
        xForward = model.A(t)*sys(n-1).xForward + chol(model.Q(t), 'lower')*randn([Nx, M]);
        
        % Calculate the particle weight
        wForward = sys(n-1).wForward.*model.likelihood(y(:, t), xForward, t);
        wForward = wForward/sum(wForward);

        % Estimate the filtered state and covariance
        sys(n).xhatFilter = xForward*wForward';
        sys(n).PhatFilter = ((ones([Nx, 1])*wForward).*(xForward-sys(n).xhatFilter*ones([1, M])))*(xForward-sys(n).xhatFilter*ones([1, M]))';
        
        % Resampling and jittering: Only when t < T
        if t < T
            % Resample if the effective sample size falls below the threshold
            ess = 1/sum(wForward.^2);
            if ess < M_T
                ir = resample(wForward);
                xForward = xForward(:, ir);
                wForward = 1/M*ones([1, M]);
            end

            % Jitter
%             C = zeros(Nx);
%             if jitter
%                 K = 0.1;
%     %             K = par.K;
%                 for i = 1:Nx
%                     E = (max(xForward(i, :))-min(xForward(i, :)));
%                     C(i, i) = K*E*M^(-1/Nx);
%                 end
%                 c = C*randn([Nx, M]);
%                 xForward = xForward + c;
% 
%     %             C = C.^2; % above, C_c is the matrix of std. devs, now we make it covariance instead because that's how we use it later on.
%             end
       end

        % Store the samples and their weights
        sys(n).xForward = xForward;
        sys(n).wForward = wForward;
        
        % Update the Prior
        sys(n).mu = model.A(t)*sys(n-1).mu;
        sys(n).Sigma = model.Q(t) + model.A(t)*sys(n-1).Sigma*model.A(t)';
    end
    
    % Remove the extra entries
    sys = sys(1:T+1);
end

%% Backward Filter
function sys = backward_filter(sys, y, model, par)

    %% Preparations
    % Get filter parameters
    M = par.M;
    M_T = par.M_T;
    
    % Get state and measurement dimenstions
    Nx = size(model.A(0), 1);
    T = size(y, 2);

    %% Initialization
    sys(T+1).xBackward = sys(T+1).xForward;
%     v = ( ...
%         model.likelihood(y(:, T), sys(T+1).xBackward, T) ...
%         .*mvnpdf(sys(T+1).xBackward.', sys(T+1).mu.', sys(T+1).Sigma.').' ...
%         ./mvnpdf(sys(T+1).xBackward.', (model.A(T)*sys(T).xForward).', model.Q(T).').' ...
%     );
    v = ( ...
        sys(T+1).wForward.*mvnpdf(sys(T+1).xBackward.', sys(T+1).mu.', sys(T+1).Sigma.').' ...
        ./(sys(T).wForward.*mvnpdf(sys(T+1).xBackward.', (model.A(T)*sys(T).xForward).', model.Q(T).').') ...
    );
    sys(T+1).wBackward = v/sum(v);

    % Initialize the backward filter using the forward particles and
    % reweigh
%     sys(T+1).xBackward = sys(T+1).xForward;
%     sys(T+1).wBackward = ( ...
%         model.likelihood(y(:, T), sys(T+1).xBackward, T) ...
%         .*mvnpdf(sys(T+1).xBackward.', sys(T+1).mu.', sys(T+1).Sigma.').' ...
%         ./sys(T+1).wForward ...
%     );
%     sys(T+1).wBackward = sys(T+1).wBackward/sum(sys(T+1).wBackward);
    
    % Calculate the smoothed weight
    predictionSum = zeros(1, M);
    for m = 1:M
        predictionSum = predictionSum ...
            + sys(T).wForward(:, m)*mvnpdf(sys(T+1).xBackward.', (model.A(T)*sys(T).xForward(:, m)).', model.Q(T).').';
    end
    wSmoothed = sys(T+1).wBackward.*predictionSum./mvnpdf(sys(T+1).xBackward.', sys(T+1).mu.', sys(T+1).Sigma.').';
    sys(T+1).wSmoothed = wSmoothed/sum(wSmoothed);
    sys(T+1).xhat = sys(T+1).xBackward*sys(T+1).wSmoothed.';

    % Old (Wrong) Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize by re-using the marginal filtering approximation at T
%     sys(T+1).xBackward = sys(T+1).xForward;
%     sys(T+1).wBackward = sys(T+1).wForward;
%     sys(T+1).wSmoothed = sys(T+1).wForward;
%     
%     % Estimate the state at T
%     sys(T+1).xhat = sys(T+1).xBackward*sys(T+1).wSmoothed.';
%     sys(T+1).Phat = sys(T+1).PhatFilter;

    %% Backward recursion
    for n = T:-1:2
        t = n-1;
        
        % Backward propagate the particles
        Sigma = (sys(n).Sigma^(-1) + model.A(t)'/model.Q(t)*model.A(t))^(-1);
%         Sigma = sys(n).Sigma - sys(n).Sigma*model.A(t)'/(model.Q(t) + model.A(t)*sys(n).Sigma*model.A(t)')*model.A(t)*sys(n).Sigma;
        mu = Sigma*( ...
            model.A(t)'*model.Q(t)*sys(n+1).xBackward ...
            + (sys(n).Sigma\sys(n).mu)*ones([1, M]) ...
        );
        xBackward = mu + chol(Sigma, 'lower')*randn([Nx, M]);
        
        % Calculate the backward particle weight
        v = sys(n+1).wBackward.*model.likelihood(y(:, t), xBackward, t);
        v = v/sum(v);
        wBackward = v;
        
        % Calculate the smoothed particle weight
        predictionSum = zeros(1, M);
        for m = 1:M
            predictionSum = predictionSum ...
                + sys(n-1).wForward(:, m)*mvnpdf(xBackward.', (model.A(t)*sys(n-1).xForward(:, m)).', model.Q(t).').';
        end
        wSmoothed = wBackward.*predictionSum./mvnpdf(xBackward.', sys(n).mu.', sys(n).Sigma.').';

        % Numerically more stable implementation (?)
%         predictionSum = zeros(1, M);
%         for m = 1:M
%             num = diag( ...
%                 (xBackward-(model.A(t)*sys(n-1).xForward(:, m))*ones([1, M]))' ...
%                 /model.Q(t) ...
%                 *(xBackward-(model.A(t)*sys(n-1).xForward(:, m))*ones([1, M])) ...
%             ).';
%             den = diag( ...
%                 (xBackward-sys(n).mu*ones([1, M]))' ...
%                 /sys(n).Sigma ...
%                 *(xBackward-sys(n).mu*ones([1, M])) ...
%             ).';
%             predictionSum = predictionSum ...
%                 + exp(log(sys(n-1).wForward(:, m)) - 1/2*(num - den));
%         end
%         wSmoothed = wBackward.*predictionSum;
        
        % Normalize
        wSmoothed = wSmoothed/sum(wSmoothed);
        
        % Estimate the smoothed state and covariance
        sys(n).xhat = xBackward*wSmoothed';
        sys(n).Phat = ((ones([Nx, 1])*wSmoothed).*(xBackward-sys(n).xhat*ones([1, M])))*(xBackward-sys(n).xhat*ones([1, M]))';
        
        % Resample
        ess = 1/sum(v.^2);
        if ess < M_T
            ir = resample(v);
            xBackward = xBackward(:, ir);
            wSmoothed = wSmoothed(:, ir);
            wBackward = 1/M*ones([1, M]);
        end
        
        % Store the particles and their weights
        sys(n).xBackward = xBackward;
        sys(n).wBackward = wBackward;
        sys(n).wSmoothed = wSmoothed;        
    end

    % Remove extra entries
    sys = sys(2:T+1);
end

%% Resampling Function
function ir = resample(w)
    M = length(w);
    ir = zeros(1, M);
    c = cumsum(w);
    u = 1/M*rand(1)+(0:M-1)'/M;
    for j = 1:M
        ir(j) = find(c > u(j), 1, 'first');
    end
end
