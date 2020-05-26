function new = makeTimeStep_implicit_new(old, n, a, b, c)
    alfa = zeros(1,n);
    gamma = zeros(1,n-1);
    new = zeros(1,n);
    
    alfa(1) = a(1);
    gamma(1) = c(1)/alfa(1);
    new(1) = old(1);%/a(1);
    
    % Forwad 
    for i = 2:n-1
        alfa(i) = a(i)-b(i)*gamma(i-1);
        gamma(i) = c(i)/alfa(i);
        new(i) = ( old(i) - b(i)*new(i-1) )/alfa(i);
    end

    % Backward
    for i = n-2:-1:1
        new(i) = new(i)-gamma(i)*new(i+1);
    end
    % hom Neumann BC
    new(n) = new(n-1);

end