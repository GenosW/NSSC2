function C_new = makeTimeStep_CrankNicolson(C, a, b, c, sim_params)
s = sim_params.s; N = sim_params.N;

%  Before
% C_iminus1 = [0,C(1:end-1)];
% C_iplus1 = [C(2:end),0];
% C_new = s*C_iminus1 + (2-2*s)*C + s*C_iplus1;

C_iminus1 = C(1:end-2);
C_iplus1 = C(3:end);
C_new = zeros(1,N);
%C_new(2:end-1) = s*C_iminus1 + (1-2*s)*C(2:end-1) + s*C_iplus1;

% NEW
C_new(2:end-1) = s*C_iminus1 + 2*(1-s)*C(2:end-1) + s*C_iplus1;
C_new(1) = C(1); % Dirichlet
C_new(end) = 2*s*C(end-1) + 2*(1-s)*C(end); % Neumann

C_new = makeTimeStep_implicit_new(C_new, N, a, b, c);
end

