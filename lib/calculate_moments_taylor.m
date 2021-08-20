function [my, Cy, Cxy] = calculate_moments_taylor(model, mx, Cx, theta)
% # Calculates the moments of the Gaussian joint approximation
% ## Usage
% * `[my, Cy, Cxy] = calculate_moments_taylor(model, mx, Cx, theta)`
%
% ## Description
% Calculates the moments `my`, `Cy`, and `Cxy` of a model of the form
%
%   x ~ N(mx, Cx)
%   y ~ p(y | x)
%
% such that
%                _ _    _  _    _         _
%              (| x |  | mx |  | Cx,   Cxy |)
%   p(x, y) ~ N(|_y_|; |_my_|, |_Cxy', Cy _|),
%
% using Taylor series expansion.
%
% ## Input
% * `model`: Model struct.
% * `mx`: Prior mean.
% * `Cx`: Prior covariance.
% * `theta`: Model parameters.
%
% ## Output
% * `my`: Mean.
% * `Cy`: Covariance.
% * `Cxy`: Cross-covariance.
%
% ## Author
% 2017-present -- Roland Hostettler

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
    
    %% Calculate moments
    my = model.py.mean(mx, theta);
    Gx = model.py.jacobian(mx, theta);
    R = model.py.cov(mx, theta);
    Cxy = Cx*Gx';
    Cy = Gx*Cx*Gx' + R;
    Cy = (Cy + Cy')/2;
end