function plotConcentrationAdvection(x, C, sim_params, C_ana, lim)

plotAna = false;
if exist('C_ana','var')
    % third parameter does not exist, so default it to something
    plotAna = true;
end
ylimits = [0 1];
if exist('lim','var')
    ylimits = lim;
end

h = sim_params.h; N = sim_params.N; numberSteps = sim_params.numberSteps;
dt = sim_params.dt; C0 = sim_params.C0; plt_title = sim_params.plt_title;

figure
plot(x, C, '-o')
hold on;
if plotAna
    plot(x, C_ana)
end
dim = [0.68 0.5 0.3 0.3];
str = {['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['C0: ' num2str(C0)]};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
grid on;
legend('Numerical Solution', 'Analytical Solution','Location','northeast')
title(plt_title)
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 h])
ylim(ylimits)
end