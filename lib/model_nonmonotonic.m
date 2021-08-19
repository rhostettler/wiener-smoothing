function model = model_nonmonotonic(Q, R, m0, P0)
% Initializes the model struct for the nonmonotonic model

% TODO:
% * Description

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
    model = model_wiener(F, Q, @g, @Gx, R, m0, P0);
end

function yp = g(x, ~)
    C = [1, 0.1, -0.49, 0.01];
    z = C*x;
    yp = ( ...
        (z < -1).*(1+z) ...
        + (z > 1).*(-1+z) ...
        + (z <= 1 & z >= -1).*(-sin(z*pi)/pi) ...
    );
end

function dgdx = Gx(x, ~)
    C = [1, 0.1, -0.49, 0.01];
    z = C*x;
    dgdx = ( ...
        (z < -1).*C ...
        + (z > 1).*C ...
        + (z <= 1 & z >= -1).*(-cos(z*pi).*C) ...
    );
end
