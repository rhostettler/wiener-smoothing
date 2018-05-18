function model = regression_model(var_q, R, m0, P0)
    alpha = [0.9, -0.8, 0.7, -0.6, 0.5];
    K = length(alpha);
    F = [
                          alpha;
        eye(K-1), zeros(K-1, 1);
    ];
    L = [1; zeros(K-1, 1)];
    Q = L*var_q*L';
    
    model = wiener_model(F, @(t) Q, @g, @(t) R, m0, P0);
    model.px.rand = @(x, t) F*x + L*sqrt(var_q)*randn(1, size(x, 2));
    model.py.fast = 0;
    model.py.logpdf = @(y, x, t) log(normpdf(y, g(x, 0, t), sqrt(R)));
end
        
function [yp, Gx, Gr] = g(x, r, t)
    C = [1 0 0 0 0];
    z = C*x;
    yp = 1/0.5*tanh(0.5*z) + r;
    Gx = (1-tanh(0.5*z).^2)*C;
    Gr = 1;
end
