function [xhat_T, P_T] = ks(y, A, B, C, Q, R, ux, Cx)
% Kalman (Rauch-Tung-Striebel/LGSS) Smoother
%
% SYNOPSIS
%   [x, P] = ks(y, A, B, C, Q, R, ux, Cx)
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
% EXAMPLE
%   See ks_example.m
% 
% SEE ALSO
%   kf, pf, ps
%
% REFERENCES
%   <references>
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@ltu.se>
%
% VERSION
%   $ ks.m $ Version 1.0 $ 2013-09-04 $
    
% TODO
%   * Implement sanity checks

    %% Parameter check
    Nx = size(A, 1);
    [Ny, T] = size(y);
    if Ny > T
        y = y.';
    end
    [Ny, T] = size(y);
    
    %% Memory Allocation
    % Filtering pass variables
    xp_t = zeros(Nx, 1, T+1);   % Stores \hat{x}_{t+1|t}
    Pp_t = zeros(Nx, Nx, T+1);  % Stores P_{t+1|t}
    Kt = zeros(Nx, Ny, T+1);    % Stores the Kalman gain
    xhat_t = zeros(Nx, 1, T+1); % Stores \hat{x}_{t|t}
    P_t = zeros(Nx, Nx, T+1);   % Stores P_{t|t}
    
    % Smoothing pass variables
    xhat_T = zeros(Nx, 1, T); % Stores \hat{x}_{t|T}
    P_T = zeros(Nx, Nx, T);   % Stores P_{t|T}
    Ct = zeros(Nx, Nx, T);    % Stores the smoothing gain
    
    %% Filtering (Forward) Pass
    % Initialization
    xhat_t(:, :, 1) = ux;
    P_t(:, :, 1) = Cx;
    y = [zeros(Ny, 1) y];
    
    % Iterations
    for t = 2:T+1
        % Prediction Update
        xp_t(:, :, t) = A*xhat_t(:, :, t-1);
        Pp_t(:, :, t) = A*P_t(:, :, t-1)*A'+B*Q*B';

        % Time Update 
        Kt(:, :, t) = Pp_t(:, :, t)*C'/(R+C*Pp_t(:, :, t)*C');
        xhat_t(:, :, t) = xp_t(:, :, t)+Kt(:, :, t)*(y(:, t)-C*xp_t(:, :, t));        
        P_t(:, :, t) = (eye(Nx)-Kt(:, :, t)*C)*Pp_t(:, :, t);
    end
    
    % Remove the initial value
    xhat_t = xhat_t(:, :, 2:T+1);
    xp_t = xp_t(:, :, 2:T+1);
    Pp_t = Pp_t(:, :, 2:T+1);
    P_t = P_t(:, :, 2:T+1);

    %% Smoothing (Backward) Pass
    % Intitialization
    xhat_T(:, :, T) = xhat_t(:, :, T);
    P_T(:, :, T) = P_t(:, :, T);
    
    % Iterations
    for t = T-1:-1:1
        Ct(:, :, t) = P_t(:, :, t)*A'/Pp_t(:, :, t+1);
        xhat_T(:, :, t) = xhat_t(:, :, t) + Ct(:, :, t)*(xhat_T(:, :, t+1) - xp_t(:, :, t+1));
        P_T(:, :, t) = P_t(:, :, t) + Ct(:, :, t)*(P_T(:, :, t+1)-Pp_t(:, :, t+1))*Ct(:, :, t)';
    end
end
