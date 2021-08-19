function [mu_y, S, B] = calculate_moments_taylor(model, x, theta)
% TODO: document

    Q = model.px.cov(x, theta);
    mu_y = model.py.mean(x, theta);
    Gx = model.py.jacobian(x, theta);
    R = model.py.cov(x, theta);

    B = Q*Gx';
    S = Gx*Q*Gx' + R;
    S = (S + S')/2;
end