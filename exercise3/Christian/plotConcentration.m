function plotConcentration(x, C, sim_params, C_ana)

plotAna = false;
if exist('C_ana','var')
    % third parameter does not exist, so default it to something
    plotAna = true;
end

h = sim_params.h; N = sim_params.N; numberSteps = sim_params.numberSteps;
dt = sim_params.dt; s = sim_params.s; plt_title = sim_params.plt_title;

figure
plot(x, C, '-o')
hold on;
if plotAna
    plot(x, C_ana)
end
dim = [0.68 0.5 0.3 0.3];
str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
grid on;
legend('Numerical Solution', 'Analytical Solution (n=100)','Location','northeast')
title(plt_title)
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 h])
ylim([0 1])
end