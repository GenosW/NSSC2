function writeFrame(v, C, x, i, sim_params, C_ana)
%WRITEFRAME Summary of this function goes here
%   Detailed explanation goes here
plotAna = false;
if ~exist('C_ana','var')
    % third parameter does not exist, so default it to something
    plotAna = true;
end
% Unpack sim_params-structure
h = sim_params.h; N = sim_params.N;
dt = sim_params.dt; dx = sim_params.dx; s = sim_params.s; 
numberSteps = sim_params.numberSteps; plt_title = sim_params.plt_title;

fig = figure;
set(fig, 'Visible', 'off');
plot(x, C, '-o')
hold on;
if plotAna
    plot(x, C_ana)
end
dim = [0.68 0.5 0.3 0.3];
str = {['step: ' i], ['dt: ' num2str(dt)], ['Steps: ' num2str(numberSteps)], ['s: ' num2str(s)]};
annotation('textbox',dim,'String',str,'FitBoxToText','on');
grid on;
legend('Numerical Solution', 'Analytical Solution (n=100)','Location','northeast')
title(plt_title)
xlabel('Distance from source [-]')
ylabel('Concentration [-]')
xlim([0 h])
ylim([0 1])

frame = getframe(gcf);
writeVideo(v,frame);
close;
end

