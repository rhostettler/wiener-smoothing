% Gaussian Approximation of the Optimal Proposal in Wiener State Space
% Systems
% 
% optimal_proposal_comparison.m -- 2016-07-05
% Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;

%% System
Q = 1;
R = 0.1;
A = 1;

mu = 0;
Sigma = 2;

g = @(x, r) x.^2 + r;
Gx = @(x, r) 2*x;
Gr = @(x, r) 1;

% g = @(x, r) cos(x) + r;
% Gx = @(x, r) -sin(x);
% Gr = @(x, r) 1;

% g = @(x, r) x + r;
% Gx = @(x, r) 1;
% Gr = @(x, r) 1;

N = 100;

%% 
xgrid = (-10:0.01:10).';
x = mu+sqrt(Sigma)*randn(1);
xsave = zeros(1, N);
ysave = zeros(1, N);
for n = 1:N
    
    %% Propagate
    q = sqrt(Q)*randn(1);
    r = sqrt(R)*randn(1);
    x_n = A*x + q;
    y_n = g(x_n, r);

    mu_n = A*mu;
    Sigma_n = A*Sigma*A' + Q;
    
    %% Illustrate
    if rem(n-1, 10) == 0
        %% Triple-Joint
        xhat_p = A*x;
        Gxn = Gx(A*mu, 0);
        Grn = Gr(A*mu, 0);
        mu_y = g(A*mu, 0);
        B = Sigma*A'*Gxn';
        C = Q*Gxn';
        D = Gxn*(A*Sigma*A'+Q)*Gxn' + Grn*R*Grn';

        yhat_p = mu_y + B'/Sigma*(x - mu);
        S = D - B'/Sigma*B;
        P = Q - C/S*C';
        xhat = xhat_p + C/S*(y_n - yhat_p);
        p_triple = normpdf(xgrid, xhat, sqrt(P));
        
        %% Conditional-Joint
        mu_y = g(A*x, 0);
        Gxn = Gx(A*x, 0);
        Grn = Gr(A*x, 0);
        C = Q*Gxn';
        S = Gxn*Q*Gxn' + Grn*R*Grn';
        
        xhat = A*x + C/S*(y_n - mu_y);
        P = Q - C/S*C';
        
        p_conditional = normpdf(xgrid, xhat, sqrt(P));
        
        %% Others
        p_likelihood = normpdf(y_n, g(xgrid, 0), sqrt(R));
        p_predict = normpdf(xgrid, A*x, sqrt(Q));
        p_target = p_likelihood.*p_predict;
        
        p_bootstrap = normpdf(xgrid, A*x, sqrt(Q));
        
        
        figure(1); clf();
        plot(xgrid, p_target/max(p_target)); hold on;
        plot(xgrid, p_bootstrap/max(p_bootstrap));
        plot(xgrid, p_likelihood/max(p_likelihood));
        plot(xgrid, p_triple/max(p_triple));
        plot(xgrid, p_conditional/max(p_conditional));
        plot([x_n, x_n], [0, 1]);
        legend('Target', 'Bootstrap', 'Likelihood', 'Triple', 'Conditional', 'True Value');
        
        pause();
    end

    %% Store
    x = x_n;
    y = y_n;
    mu = mu_n;
    Sigma = Sigma_n;
    
    xsave(:, n) = x;
    ysave(:, n) = y;
   
end






