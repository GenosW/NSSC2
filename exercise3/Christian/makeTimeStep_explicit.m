function C_new = makeTimeStep_explicit(C, s, mode)
   
    if (mode==1)
        l = length(C);
        C_new = zeros(1,l);
        C_iminus1 = C(1:end-2);
        C_i = C(2:end-1);
        C_iplus1 = C(3:end);
        C_temp = C_i + s*(C_iminus1 - 2*C_i + C_iplus1);
        C_new(1) = C(1);
        C_new(2:end-1) = C_temp;
        C_new(end) = C_new(end-1);
    else if (mode==2)
        l = length(C);
        C_new = zeros(1,l);
        C_iminus1 = C(1:end-2);
        C_i = C(2:end-1);
        C_iplus1 = C(3:end);
        C_temp = C_i + s*(C_iminus1 - 2*C_i + C_iplus1);
        C_new(1) = C(1);
        C_new(2:end-1) = C_temp;
        C_new(end) = C(end);
    end

end
