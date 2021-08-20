function model = model_nonmonotonic(Q, R, m0, P0)
% # Initializes the model struct for the nonmonotonic model
% ## Usage
% * `model = model_nonmonotonic(Q, R, m0, P0)`
%
% ## Description
% Non-monotonic model used in the example in [1].
%
% ## Input
% * `Q`: Process noise covariance.
% * `R`: Measurement noise covariance.
% * `m0`: Initial state mean.
% * `P0`: Initial state covariance.
%
% ## Output
% * `model`: Model struct.
% 
% ## References
% * R. Hostettler and T. B. Schön, “Auxiliary-particle-filter-based two-
%   filter smoothing for Wiener state-space models,” in 21th International 
%   Conference on Information Fusion (FUSION), Cambridge, UK, July 2018
% 
% ## Authors
% * 2017-present -- Roland Hostettler

%{
% This file is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by thee 
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
% 
% This file is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
% FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
% more details.
% 
% You should have received a copy of the GNU General Public License along 
% with this file. If not, see <http://www.gnu.org/licenses/>.
%}

    %% Defaults
    narginchk(4, 4);

    %% Dynamics
    % den = [1, 0.368, 0.888, 0.524, 0.555];
    % num = [0, 1, 0.1, -0.49, 0.01];
    % F = tf(num, den, 1, 'Variable', 'z^-1');
    % F = ss(F);
    F = [
        -0.3680, -0.8880, -0.5240, -0.5550;
              1,       0,       0,       0;
              0,       1,       0,       0;
              0,       0,       1,       0;
    ];
    model = model_wiener(F, Q, @g, @Gx, R, m0, P0);
end

%% Measurement function
function yp = g(x, ~)
    C = [1, 0.1, -0.49, 0.01];
    z = C*x;
    yp = ( ...
        (z < -1).*(1+z) ...
        + (z > 1).*(-1+z) ...
        + (z <= 1 & z >= -1).*(-sin(z*pi)/pi) ...
    );
end

%% Jacobian of the measurement function
function dgdx = Gx(x, ~)
    C = [1, 0.1, -0.49, 0.01];
    z = C*x;
    dgdx = ( ...
        (z < -1).*C ...
        + (z > 1).*C ...
        + (z <= 1 & z >= -1).*(-cos(z*pi).*C) ...
    );
end
