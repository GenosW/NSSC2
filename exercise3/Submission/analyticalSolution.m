function C = analyticalSolution(N,n,t)

    x = linspace(0,1,N);
    sum = zeros(1,N);
    for i = 0:n
        sum = sum + (-1)^i/((i+0.5)*pi).*cos((i+0.5)*pi.*x).*exp(-(i+0.5).^2*pi^2*t);
    end
    C = ones(1,N) -2*sum;
    C = flip(C);

end