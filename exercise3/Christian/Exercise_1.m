%% Exercise 1.1

clear all;
clc;
close all;

h = 5;
N = 100;
dt = 0.00005;
numberSteps = 1000;
makeVideo = false;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C(1) = 1;
v = VideoWriter('concentration_1_1');
open(v);

for i = 1:numberSteps
    C = makeTimeStep_explicit(C,s,1);
    
    if makeVideo
        C_analytic = analyticalSolution(N,1000,i*dt);
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C)
        hold on;
        plot(linspace(0,1,N), C_analytic, 'o')
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

close(v);

C_analytic = analyticalSolution(N,1000,numberSteps*dt);
figure
plot(linspace(0,1,N), C)
hold on;
plot(linspace(0,1,N), C_analytic, 'o')
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
dt = 0.00005;
numberSteps = 10000;
makeVideo = false;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C(1) = 1;
v = VideoWriter('concentration_1_2');
open(v);

for i = 1:numberSteps
    C = makeTimeStep_explicit(C,s,2);
    
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C)
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

close(v);

figure
plot(linspace(0,1,N), C)
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
numberSteps = 1000;
makeVideo = false;

dx = 1/(N-1);
s = dt/dx^2;

C = zeros(1,N);
C_check = zeros(N,1);
C(1) = 1;
C_check(1) = 1;
v = VideoWriter('concentration_1_2');
open(v);
M = zeros(N);
M(  1:1+N:N*N) = 1+2*s;
M(N+1:1+N:N*N) = -s;
M(  2:1+N:N*N-N) = -s;
for i = 1:numberSteps
    C = makeTimeStep_implicit(C,ones(1,N)*(1+2*s), ones(1,N)*(-s), ones(1,N)*(-s));
    C_check = M\C_check;
    C_check(1) = 1;
    C_check(end) = C_check(end-1);
    if makeVideo
        h = figure;
        set(h, 'Visible', 'off');
        plot(linspace(0,1,N), C)
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

close(v);

figure
plot(linspace(0,1,N), C)
hold on;
plot(linspace(0,1,N),C_check, 'o')
title('numerical solution 1.3')
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 1])
ylim([0 1])