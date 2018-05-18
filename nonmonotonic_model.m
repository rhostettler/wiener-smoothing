function model = nonmonotonic_model(Q, R, m0, P0)

    %% Dynamics
%     den = [1, 0.368, 0.888, 0.524, 0.555];
%     num = [0, 1, 0.1, -0.49, 0.01];
%     F = tf(num, den, 1, 'Variable', 'z^-1');
%     F = ss(F);
    F = [
        -0.3680, -0.8880, -0.5240, -0.5550;
              1,       0,       0,       0;
              0,       1,       0,       0;
              0,       0,       1,       0;
    ];
    model = wiener_model(F, Q, @g, @(t) R, m0, P0);
    model.py.fast = 0;
    model.py.logpdf = @(y, x, t) log(normpdf(y, g(x, 0, t), sqrt(R)));
end

function [yp, Gx, Gr] = g(x, r, t)
    C = [1, 0.1, -0.49, 0.01];
    z = C*x;
    yp = ( ...
        (z < -1).*(1+z) ...
        + (z > 1).*(-1+z) ...
        + (z <= 1 & z >= -1).*(-sin(z*pi)/pi) ...
    ) + r;
    Gx = ( ...
        (z < -1).*C ...
        + (z > 1).*C ...
        + (z <= 1 & z >= -1).*(-cos(z*pi).*C) ...
    );
    Gr = 1;
end
