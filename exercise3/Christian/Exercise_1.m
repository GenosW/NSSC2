%% Exercise 1.1

clear all;
clc;
close all;

h = 5;
N = 100;
dt = 0.00004;
numberSteps = 10000;
makeVideo = true;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C(1) = 1;
if makeVideo
    v = VideoWriter('concentration_1_1');
    open(v);
end

for i = 1:numberSteps
    C = makeTimeStep_explicit(C,s,1);
    
    if makeVideo & mod(i,30)==0
        C_analytic = analyticalSolution(N,100,i*dt);
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, '-o')
        hold on;
        plot(linspace(0,1,N), C_analytic)
        dim = [0.68 0.5 0.3 0.3];
        str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
        annotation('textbox',dim,'String',str,'FitBoxToText','on');
        grid on;
        legend('Numerical Solution', 'Analytical Solution (n=100)','Location','northeast')
        title('numerical vs analytical solution 1.1')
        xlabel('Distance from source [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([0 1])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

C_analytic = analyticalSolution(N,100,numberSteps*dt);
figure
plot(linspace(0,1,N), C, '-o')
hold on;
plot(linspace(0,1,N), C_analytic)
dim = [0.68 0.5 0.3 0.3];
str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
grid on;
legend('Numerical Solution', 'Analytical Solution (n=100)','Location','northeast')
title('numerical vs analytical solution 1.1')
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([0 1])


%% Exercise 1.2

clear all;
clc;
close all;

h = 5;
N = 100;
dt = 0.00004;
numberSteps = 10000;
makeVideo = true;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C(1) = 1;
if makeVideo
    v = VideoWriter('concentration_1_2');
    open(v);
end

for i = 1:numberSteps
    C = makeTimeStep_explicit(C,s,2);
    
    if makeVideo & mod(i,30)==0
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, '-o')
        hold on;
        dim = [0.68 0.5 0.3 0.3];
        str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
        annotation('textbox',dim,'String',str,'FitBoxToText','on');
        grid on;
        legend('Numerical Solution', 'Location', 'northeast')
        title('numerical solution 1.2')
        xlabel('Distance from source [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([0 1])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

figure
plot(linspace(0,1,N), C, '-o')
hold on;
dim = [0.68 0.5 0.3 0.3];
str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
grid on;
legend('Numerical Solution', 'Location', 'northeast')
title('numerical solution 1.2')
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([0 1])

%% Exercise 1.3

clear all;
clc;
close all;

h = 5;
N = 100;
dt = 0.00005;
numberSteps = 10000;
makeVideo = false;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
% C_check = zeros(N,1);
C(1) = 1;
% C_check(1) = 1;
if makeVideo
    v = VideoWriter('concentration_1_3');
    open(v);
end
% M = zeros(N);
% M(  1:1+N:N*N) = 1+2*s;
% M(N+1:1+N:N*N) = -s;
% M(  2:1+N:N*N-N) = -s;
% M(N-1,N-1) = 1+s;
% M(1,1) = 1;
% M(1,2) = 0;
% M = M(1:N-1,1:N-1);
for i = 1:numberSteps
    C = makeTimeStep_implicit(C,s);
%     C_check(1:end-1) = M\C_check(1:end-1);
%     C_check(end) = C_check(end-1);
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, 'LineWidth', 2)
        grid on;
        title('numerical solution 1.3')
        xlabel('Distance from source [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([0 1])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

figure
plot(linspace(0,1,N), C)
hold on;
% plot(linspace(0,1,N),C_check, 'o')
title('numerical solution 1.3')
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([0 1])


%% Exercise 1.4

clear all;
clc;
close all;

h = 5;
N = 100;
dt = 0.00005;
numberSteps = 10000;
makeVideo = false;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C_check = zeros(N,1);
C(1) = 1;
C_check(1) = 1;
if makeVideo
    v = VideoWriter('concentration_1_4');
    open(v);
end
M = zeros(N);
M(  1:1+N:N*N) = 1+2*s;
M(N+1:1+N:N*N) = -s;
M(  2:1+N:N*N-N) = -s;
M(N-1,N-1) = 1+s;
M(1,1) = 1;
M(1,2) = 0;
M = M(1:N-1,1:N-1);

for i = 1:numberSteps
    C = makeTimeStep_implicit_CrankNicolson(C,s);
    C_check_iminus1 = [C_check;0];
    C_check_iplus1 = [0;C_check];
    C_check = s*C_check_iminus1(1:end-1) + (1-2*s)*C_check + s*C_check_iplus1(2:end);
    C_check(1:end-1) = M\C_check(1:end-1);
    C_check(end) = C_check(end-1);
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, 'LineWidth', 2)
        grid on;
        title('numerical solution 1.4')
        xlabel('Distance from source [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([0 1])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

figure
plot(linspace(0,1,N), C)
hold on;
plot(linspace(0,1,N),C_check, 'o')
title('numerical solution 1.4')
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([0 1])

%% Exercise 2.1

clear all;
clc;
close all;

h = 5;
N = 101;
dt = 0.01;
numberSteps = 10;
makeVideo = false;

dx = 1/(N-1);

C = zeros(1,N);
for i = 0:1:100
    if (i > 10 & i < 30)
        C(i) = 1;
    end
end

if makeVideo
    v = VideoWriter('concentration_2_1_rectangle');
    open(v);
end

C0 = dt/dx;
for i = 1:numberSteps
    C = makeTimeStep_upwind(C,C0);
    
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, 'LineWidth',2)
        grid on;
        title('numerical solution 2.1 Rectangle')
        xlabel('Distance [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([-2 2])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

figure
plot(linspace(0,1,N), C, 'LineWidth',2)
grid on;
title('numerical solution 2.1')
xlabel('Distance [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([-2 2])


% part two

t = linspace(0,1,N);
C = exp(-10.*(4.*t-1).^2);

if makeVideo
    v = VideoWriter('concentration_2_1_gauss');
    open(v);
end

C0 = dt/dx;
for i = 1:numberSteps
    C = makeTimeStep_upwind(C,C0);
    
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C, 'LineWidth',2)
        grid on;
        title('numerical solution 2.1 Gauss')
        xlabel('Distance [-]')
        ylabel('Concentration [-]')
        xlim([0 1])
        ylim([-2 2])

        frame = getframe(gcf);
        writeVideo(v,frame);
        close;
    end
    
end

try
close(v);
end

figure
plot(linspace(0,1,N), C, 'LineWidth',2)
grid on;
title('numerical solution 2.1')
xlabel('Distance [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([-2 2])