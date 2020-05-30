function ana = analyticalSolutionAdvection(C,k)

for i =1:length(C)
    try
        ana(i) = C(i-k);
    catch
        ana(i) = 0;
    end
end

end