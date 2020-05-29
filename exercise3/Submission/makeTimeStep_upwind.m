function C_new = makeTimeStep_upwind(C, C0)

C_new = zeros(1,length(C));
C_new(2:end) = C(2:end).*(1-C0) + C(1:end-1).*C0;
C_new(1) = C(1)*(1+C0)-C(2)*C0;

end