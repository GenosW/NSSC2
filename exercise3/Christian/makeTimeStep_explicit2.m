function C_new = makeTimeStep_explicit2(C, s, bc_left, bc_right)
% Reserve new vector
l = length(C);
C_new = zeros(1,l);

% Some aliasing
C_iminus1 = C(1:end-2);
C_i = C(2:end-1);
C_iplus1 = C(3:end);

% Perform udpate for inner
C_new(2:end-1) = C_i + s*(C_iminus1 - 2*C_i + C_iplus1);

% Update on boundary
if (bc_left==0)
    C_new(1) = C(1);
elseif (bc_left==1)
    C_new(1) = C(2);
end
if (bc_right==0) % Dirichlet
    C_new(end) = C(end);
elseif (bc_right==1) % Neumann
    C_new(end) = C_new(end-1);
end
end

