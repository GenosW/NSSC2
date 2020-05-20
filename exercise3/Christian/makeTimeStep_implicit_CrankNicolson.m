function C_new = makeTimeStep_implicit_CrankNicolson(C, s)

    C_iminus1 = [C,0];
    C_iplus1 = [0,C];
    C = s*C_iminus1(1:end-1) + (1-2*s)*C + s*C_iplus1(2:end);
    
    N = length(C);
    a = ones(1,N-1)*(1+2*s);
    a(1) = 1;
    a(end) = 1+s;
    b = ones(1,N-1)*(-s);
    c = ones(1,N-1)*(-s);
    c(1) = 0;
    
    n = N-1;
    alfa = zeros(1,n);
    gamma = zeros(1,n);
    v = zeros(1,n);
    C_new = zeros(1,n+1);
    
    alfa(1) = a(1);
    gamma(1) = c(1)/a(1);
    v(1) = C(1)/a(1);
    
    for i = 2:n
        alfa(i) = a(i)-b(i)*gamma(i-1);
        gamma(i) = c(i)/alfa(i);
        v(i) = (C(i)-b(i)*v(i-1))/alfa(i);
    end
    
    C_new(n) = v(n);
    
    for i = n-1:-1:1
        C_new(i) = v(i)-gamma(i)*C_new(i+1);
    end
    
    C_new(end) = C_new(end-1);

end