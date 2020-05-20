function C_new = makeTimeStep_implicit(C, a, b, c)

    n = length(C)-1;
    alfa = zeros(1,n);
    gamma = zeros(1,n);
    v = zeros(1,n);
    C_new = zeros(1,n+1);
    
    alfa(1) = a(1);
    gamma(1) = c(1)/a(1);
    v(1) = C(1)/a(1);
    
    for i = 2:n-1
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