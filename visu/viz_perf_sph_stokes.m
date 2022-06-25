clear

% load scaling_data
fid = fopen('../out_perf/out_SphericalStokes_mtp_daint.txt','r');  stokes_3D_daint  = fscanf(fid, '%d %d %d %d %f %f %f %f', [8 Inf]); fclose(fid); % nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../out_perf/out_SphericalStokes_mtp_ampere.txt','r'); stokes_3D_ampere = fscanf(fid, '%d %d %d %d %f %f %f %f', [8 Inf]); fclose(fid); % nx ny nz ittot t_toc A_eff t_it T_eff

fid = fopen('../out_perf/out_SphericalStokes_pareff_daint.txt','r');  stokes_3D_mxpu_daint  = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../out_perf/out_SphericalStokes_pareff_ampere.txt','r'); stokes_3D_mxpu_ampere = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff

nrep1 = 3; % number of repetitions of the experiment
nrep2 = 5; % number of repetitions of the experiment
nrep3 = 4; % number of repetitions of the experiment
my_type = "my_max";
% my_type = "my_mean";

stokes_3D_daint_2  = average_exp(stokes_3D_daint, nrep1, my_type);
stokes_3D_ampere_2 = average_exp(stokes_3D_ampere, nrep2, my_type);
stokes_3D_mxpu_daint_2  = average_exp(stokes_3D_mxpu_daint, nrep3, my_type);
stokes_3D_mxpu_ampere_2 = average_exp(stokes_3D_mxpu_ampere, nrep2, my_type);

% T_peak_volta = 840;
% T_peak_octo  = 254;
T_peak_daint = 561;
T_peak_ampere = 1355;

% no hide_comm perfs
single_daint_stokes = 78.1;
% single_volta_stokes = 334.1;
single_ampere_stokes = 148.0;
sc = 100; % to get percent

FS = 20;
mylim = [0 1400];
mylimx = [32 550];
ylab = 1270;

mylim2  = [0.945 1.003].*sc;
mylimx2 = [0.8 3.e3];
ylab2 = 0.963.*sc;

fig1 = 1;
fig2 = 1;
do_print = 0;
%%
if fig1==1
figure(1),clf,set(gcf,'color','white','pos',[1400 10 500 400])
semilogx(stokes_3D_ampere_2(2,:),stokes_3D_ampere_2(end,:), '-o', ...
         stokes_3D_daint_2(2,:),stokes_3D_daint_2(end,:), '-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), set(gca, 'fontsize',FS, 'linewidth',1.4)
hold on
semilogx(stokes_3D_ampere_2(2,:),T_peak_ampere*ones(size(stokes_3D_ampere_2(2,:))),'k:', ...
         stokes_3D_ampere_2(2,:),T_peak_daint*ones(size(stokes_3D_ampere_2(2,:))), 'k--',...
         'linewidth',1.5, 'MarkerFaceColor','k')
hold off
title({'3D spherical Stokes'},'fontsize',FS-2)
lg=legend('Tesla A100 SXM4', 'Tesla P100 PCIe'); set(lg,'box','off')
ylim(mylim)
xlim(mylimx)
ylabel({' ';'\bf{T_{eff} [GB/s]}'}, 'fontsize',FS)
set(gca, 'XTick',stokes_3D_ampere_2(2,:))
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{nx}', 'fontsize',FS)
% text(33,ylab,'(b)','fontsize',FS+2,'fontname','Courier')

% pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*0.97  pos1(2)*2 pos1(3)*1.12 pos1(4)*0.8])
% pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.97  pos2(2)*2 pos2(3)*1.12 pos2(4)*0.8])
fig = gcf;
fig.PaperPositionMode = 'auto';
if do_print==1, print('fig_perf_mtp3D','-dpng','-r300'); end
end
%%
if fig2==1
figure(2),clf,set(gcf,'color','white','pos',[1400 10 500 400])
semilogx(stokes_3D_mxpu_ampere_2(1,:),stokes_3D_mxpu_ampere_2(end,:)./single_ampere_stokes.*sc, '-o', ...
         stokes_3D_mxpu_daint_2(1,:),stokes_3D_mxpu_daint_2(end,:)./single_daint_stokes.*sc, '-o', ...
     'linewidth',3, 'MarkerFaceColor','k'), set(gca, 'fontsize',FS, 'linewidth',1.4)
title({'3D spherical Stokes'},'fontsize',FS-2)
% ylabel({' ';'\bf{E}'}, 'fontsize',FS)
lg=legend('Tesla A100 SXM4', 'Tesla P100 PCIe'); set(lg,'box','off')
ylim(mylim2)
xlim(mylimx2)
ylabel({' ';'\bf{E}'}, 'fontsize',FS)
ytickformat('%g\%')
set(gca, 'XTick',stokes_3D_mxpu_daint_2(1,:))
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{P (GPUs)}', 'fontsize',FS)
% text(1.1,ylab2,'(b)','fontsize',FS+2,'fontname','Courier')

% pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*1.04  pos1(2)*2 pos1(3)*1.1 pos1(4)*0.8])
% pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.96  pos2(2)*2 pos2(3)*1.1 pos2(4)*0.8])

fig = gcf;
fig.PaperPositionMode = 'auto';
if do_print==1, print('fig_parperf3D','-dpng','-r300'); end

end

%%% support function
function B = average_exp(A, nrep, type)

nexp = size(A,2)/nrep;
B    = zeros(size(A,1),nexp);

if type == "my_mean"
    for i=1:nexp
        B(:,i) = mean(A(:,(i-1)*nrep+1:i*nrep),2);
    end
elseif type == "my_max"
    for i=1:nexp
        B(:,i) = max(A(:,(i-1)*nrep+1:i*nrep),[],2);
    end
end
end
