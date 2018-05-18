function [ms, Ps] = rtss(y, A, C, Q, R, m0, P0)
% Rauch-Tung-Striebel Smoother
%
% USAGE
%   [ms, Ps] = RTSS(y, A, B, C, Q, R, m0, P0)
% 
% DESCRIPTION
%   State estimation smoother for a linear Gaussian state space system
%   (LGSS), also known as Kalman smoother or Rauch–Tung–Striebel smoother.
%
%   The LGSS is defined as
%
%       x(t+1) = A x(t) + B v(t)
%         y(t) = C x(t) + w(t)
%
%   for t = 1, ..., T and
%   
%       v(t) ~ N(0, Q), w(t) = N(0, R), x(0) = N(ux, Cx)
%
% AUTHORS
%   2013-09-04 -- Roland Hostettler <roland.hostettler@ltu.se>
%   2018-05-18 -- Roland Hostettler <roland.hostettler@aalto.fi>

    %% Parameter check
    narginchk(7, 7);
    Nx = size(A, 1);
    [Ny, T] = size(y);
    if Ny > T
        y = y.';
    end
    [Ny, T] = size(y);
    
    %% Memory Allocation
    % Filtering pass variables
    mp = zeros(Nx, 1, T+1);   % Stores \hat{x}_{t+1|t}
    Pp = zeros(Nx, Nx, T+1);  % Stores P_{t+1|t}
    Kt = zeros(Nx, Ny, T+1);    % Stores the Kalman gain
    m = zeros(Nx, 1, T+1); % Stores \hat{x}_{t|t}
    P = zeros(Nx, Nx, T+1);   % Stores P_{t|t}
    
    % Smoothing pass variables
    ms = zeros(Nx, 1, T); % Stores \hat{x}_{t|T}
    Ps = zeros(Nx, Nx, T);   % Stores P_{t|T}
    Ct = zeros(Nx, Nx, T);    % Stores the smoothing gain
    
    %% Filtering (Forward) Pass
    % Initialization
    m(:, :, 1) = m0;
    P(:, :, 1) = P0;
    y = [zeros(Ny, 1) y];
    
    % Iterations
    for t = 2:T+1
        % Prediction Update
        mp(:, :, t) = A*m(:, :, t-1);
        Pp(:, :, t) = A*P(:, :, t-1)*A' + Q;

        % Time Update 
        Kt(:, :, t) = Pp(:, :, t)*C'/(R+C*Pp(:, :, t)*C');
        m(:, :, t) = mp(:, :, t)+Kt(:, :, t)*(y(:, t)-C*mp(:, :, t));        
        P(:, :, t) = (eye(Nx)-Kt(:, :, t)*C)*Pp(:, :, t);
    end
    
    % Remove the initial value
    m = m(:, :, 2:T+1);
    mp = mp(:, :, 2:T+1);
    Pp = Pp(:, :, 2:T+1);
    P = P(:, :, 2:T+1);

    %% Smoothing (Backward) Pass
    % Intitialization
    ms(:, :, T) = m(:, :, T);
    Ps(:, :, T) = P(:, :, T);
    
    % Iterations
    for t = T-1:-1:1
        Ct(:, :, t) = P(:, :, t)*A'/Pp(:, :, t+1);
        ms(:, :, t) = m(:, :, t) + Ct(:, :, t)*(ms(:, :, t+1) - mp(:, :, t+1));
        Ps(:, :, t) = P(:, :, t) + Ct(:, :, t)*(Ps(:, :, t+1)-Pp(:, :, t+1))*Ct(:, :, t)';
    end
end
