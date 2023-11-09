clear

% ===================================================================================================
% FIGURES PARAMETERS
% ===================================================================================================
Save    = 0;
paper   = 1;
equal   = 0;
font    = 22;
font_ax = 18;


% ===================================================================================================
% DATA LOADING
% ===================================================================================================
DAT    = 'double';
% CARTESIAN COORDINATES -----------------------------------------------------------------------------
folder = '../out_cart';

A             =load([folder '/1_infos.inf']);
nx_CART       = A(5);
ny_CART       = A(6);
nz_CART       = A(7);

A             =load([folder '/1_phys.inf']);
rho0_CART     = A(3);
rho_in_CART   = A(4);
g_CART        = A(8);
r_CART        = A(10);

name          = [folder '/A_X3D.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nx_CART,ny_CART,nz_CART);
X3D_CART      = B;

name          = [folder '/A_Y3D.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nx_CART,ny_CART,nz_CART);
Y3D_CART      = B;

name          = [folder '/A_Z3D.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nx_CART,ny_CART,nz_CART);
Z3D_CART      = B;

name          = [folder '/A_P.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nx_CART,ny_CART,nz_CART);
P_CART        = B;

name          = [folder '/A_TAU_ZZ.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nx_CART,ny_CART,nz_CART);
T_ZZ_CART     = B;


% CYLINDRICAL COORDINATES ---------------------------------------------------------------------------
folder = '../out_cyl';
rco_CYL = 1000;

A             =load([folder '/1_infos.inf']);
nr_CYL        = A(5);
nth_CYL       = A(6);
nz_CYL        = A(7);

A             =load([folder '/1_phys.inf']);
rho0_CYL      = A(3);
rho_in_CYL    = A(4);
g_CYL         = A(8);
r_CYL         = A(11);

name          = [folder '/A_R.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
R_CYL         = B;

name          = [folder '/A_TH.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
TH_CYL        = B;

name          = [folder '/A_Z.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
Z_CYL         = B;

name          = [folder '/A_P.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
P_CYL         = B;

name          = [folder '/A_D_RR.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
D_RR_CYL      = B;

name          = [folder '/A_TAU_RR.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_CYL,nth_CYL,nz_CYL);
T_RR_CYL      = B;

% SPHERICAL COORDINATES -----------------------------------------------------------------------------
folder = '../out_sph';
rco_SPH = 1000;

A             =load([folder '/1_infos.inf']);
lr_SPH        = A(2);
nr_SPH        = A(5);
nth_SPH       = A(6);
nph_SPH       = A(7);

A             =load([folder '/1_phys.inf']);
rho0_SPH      = A(3);
rho_in_SPH    = A(4);
g_SPH         = A(8);
r_SPH         = A(11);

name          = [folder '/A_R.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH,nth_SPH,nph_SPH);
R_SPH         = B;

name          = [folder '/A_TH.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH,nth_SPH,nph_SPH);
TH_SPH        = B;

name          = [folder '/A_PH.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH,nth_SPH,nph_SPH);
PH_SPH        = B;

name          = [folder '/A_P.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH,nth_SPH,nph_SPH);
P_SPH         = B;

name          = [folder '/A_TAU_RR.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH,nth_SPH,nph_SPH);
T_RR_SPH      = B;

name          = [folder '/A_VR.bin'];
fid           = fopen(name);
A             = fread(fid, DAT);
B             = reshape(A,nr_SPH+1,nth_SPH,nph_SPH);
VR_SPH        = B;

% ===================================================================================================
% COORDINATE TRANSFORMATION
% ===================================================================================================
% CYLINDRICAL COORDINATES ---------------------------------------------------------------------------
[X_cCYL, Y_cCYL, Z_cCYL] = pol2cart(TH_CYL, R_CYL, Z_CYL);

% SPHERICAL COORDINATES -----------------------------------------------------------------------------
[X_cSPH,Y_cSPH,Z_cSPH] = sph2cart(PH_SPH,pi/2-TH_SPH,R_SPH);
% CORRESPONDANCE
% X_CART(:,1,1) = R
% Y_CART(1,1,:) = PHI
% Z_CART(1,:,1) = THETA

% ===================================================================================================
% TOTAL STRESS CALCULATION
% ===================================================================================================
S_ZZ_CART = -P_CART + T_ZZ_CART;
S_RR_CYL  = -P_CYL  + T_RR_CYL;
S_RR_SPH  = -P_SPH  + T_RR_SPH;

% ===================================================================================================
% NORMALIZING STRESSES
% ===================================================================================================
BUO_CART = (rho0_CART-rho_in_CART)*g_CART*r_CART;
BUO_CYL  = (rho0_CYL -rho_in_CYL )*g_CYL *r_CYL;
BUO_SPH  = (rho0_SPH -rho_in_SPH )*g_SPH *r_SPH;

% ===================================================================================================
% FIGURES
% ===================================================================================================
figure(1),colormap(turbo),clf
% CARTESIAN COORDINATES -----------------------------------------------------------------------------
subplot('Position',[0.06 0.67 0.25 0.27])
pcolor(squeeze(Y3D_CART(fix(end/2),:,:))/r_CART,squeeze(Z3D_CART(fix(end/2),:,:))/r_CART,squeeze(S_ZZ_CART(fix(end/2),:,:))/BUO_CART)
colorbar
axis image,shading interp
ylabel('\bf{Z/R}'),t=title('Cartesian');
if paper == 1
    clim([-1,1])
    if equal==1
        clim([-1 1])
    end
    set(gca,'XTick',[],'FontSize',font_ax,'FontName','Courier')
    colorbar off
    set(t,'Position',[0 3.27 0],'FontSize',font,'FontWeight','Normal')
end
text(-2.7,2.45,'a)','FontSize',font,'FontName','Courier')

subplot('Position',[0.06 0.37 0.25 0.27])
pcolor(squeeze(Y3D_CART(fix(end/2),:,:))/r_CART,squeeze(Z3D_CART(fix(end/2),:,:))/r_CART,squeeze(P_CART(fix(end/2),:,:))/BUO_CART)
colorbar
axis image,shading interp
% xlabel('Y'),
ylabel('\bf{Z/R}'),t4=title('\bf{P/(\Delta\rhogR)}');
if paper == 1
    clim([-0.8,0.8])
    if equal==1
        clim([-1 1])
    end
    set(gca,'XTick',[],'FontSize',font_ax,'FontName','Courier')
    colorbar off
    set(t4,'Position',[19.8 0 0],'FontSize',font,'Rotation',90,'FontWeight','Normal')
end
text(-2.7,2.45,'b)','FontSize',font,'FontName','Courier')

subplot('Position',[0.06 0.07 0.25 0.27])
pcolor(squeeze(Y3D_CART(fix(end/2),:,:))/r_CART,squeeze(Z3D_CART(fix(end/2),:,:))/r_CART,squeeze(T_ZZ_CART(fix(end/2),:,:))/BUO_CART)
colorbar
axis image,shading interp
xlabel('\bf{Y/R}'),ylabel('\bf{Z/R}')
t5=title('\bf{\tau_{vert}/(\Delta\rhogR)}');
if paper == 1
    clim([-0.2,0.2])
    if equal==1
        clim([-1 1])
    end
    set(gca,'FontSize',font_ax,'FontName','Courier')
    colorbar off
    set(t5,'Position',[20 0 0],'FontSize',font,'Rotation',90,'FontWeight','Normal')
end
text(-2.7,2.45,'c)','FontSize',font,'FontName','Courier')


% CYLINDRICAL COORDINATES ---------------------------------------------------------------------------
subplot('Position',[0.375 0.67 0.25 0.27])
pcolor(squeeze( Y_cCYL(:,:,fix(end/2)))/r_CYL,(squeeze(X_cCYL(:,:,fix(end/2)))-rco_CYL)/r_CYL,squeeze(S_RR_CYL(:,:,fix(end/2)))/BUO_CYL)
colorbar
axis image,shading interp
ylabel('\bf{r/R}'),t2=title('Cylindrical');
if paper == 1
    clim([-1,1])
    if equal==1
        clim([-1 1])
    end
    set(gca,'XTick',[],'FontSize',font_ax,'FontName','Courier')
    colorbar off
    set(t2,'Position',[0 3.3 0],'FontSize',font,'FontWeight','Normal')
end
text(-2.7,2.45,'d)','FontSize',font,'FontName','Courier')

subplot('Position',[0.375 0.37 0.25 0.27])
pcolor(squeeze( Y_cCYL(:,:,fix(end/2)))/r_CYL,(squeeze(X_cCYL(:,:,fix(end/2)))-rco_CYL)/r_CYL,squeeze(P_CYL(:,:,fix(end/2)))/BUO_CYL),hold on
colorbar
axis image,shading interp
ylabel('\bf{r/R}')
if paper == 1
    clim([-0.8,0.8])
    if equal==1
        clim([-1 1])
    end
    set(gca,'XTick',[],'FontSize',font_ax,'FontName','Courier')
    colorbar off
end
text(-2.7,2.45,'e)','FontSize',font,'FontName','Courier')

subplot('Position',[0.375 0.07 0.25 0.27])
pcolor(squeeze( Y_cCYL(:,:,fix(end/2)))/r_CYL,(squeeze(X_cCYL(:,:,fix(end/2)))-rco_CYL)/r_CYL,squeeze(T_RR_CYL(:,:,fix(end/2)))/BUO_CYL),hold on
colorbar
axis image,shading interp,xlabel('\bf{\theta/R}'),ylabel('\bf{r/R}')
t5=title('\bf{\sigma_{vert}/(\Delta\rhogR)}');
if paper == 1
    clim([-0.2,0.2])
    if equal==1
        clim([-1 1])
    end
    set(gca,'FontSize',font_ax,'FontName','Courier')
    colorbar off
    set(t5,'Position',[12.3 13.3 0],'Rotation',90,'FontSize',font,'FontWeight','Normal')
end
text(-2.7,2.45,'f)','FontSize',font,'FontName','Courier')


% SPHERICAL COORDINATES -----------------------------------------------------------------------------
subplot('Position',[0.63 0.67 0.25 0.27])
pcolor(squeeze(Z_cSPH(:,:,fix(end/2)))/r_SPH,(squeeze(X_cSPH(:,:,fix(end/2)))-rco_SPH)/r_SPH,squeeze(S_RR_SPH(:,:,fix(end/2)))/BUO_SPH)
c=colorbar;
axis image,shading interp
t3=title('Spherical');
if paper == 1
    clim([-1,1])
    if equal==1
        clim([-1 1])
    end
    set(c,'Position',[0.885 0.675 0.02 0.26],'FontSize',font_ax)
    set(gca,'XTick',[],'YTick',[],'FontName','Courier')
    set(t3,'Position',[0 3.3 -7.1054e-15],'FontSize',font,'FontWeight','Normal')
end
text(-2.7,2.45,'g)','FontSize',font,'FontName','Courier')

subplot('Position',[0.63 0.37 0.25 0.27])
pcolor(squeeze(Z_cSPH(:,:,fix(end/2)))/r_SPH,(squeeze(X_cSPH(:,:,fix(end/2)))-rco_SPH)/r_SPH,squeeze(P_SPH(:,:,fix(end/2)))/BUO_SPH)
d=colorbar;
axis image,shading interp
if paper == 1
    clim([-0.8,0.8])
    if equal==1
        clim([-1 1])
    end
    set(d,'Position',[0.885 0.375 0.02 0.26],'FontSize',font_ax)
    set(gca,'XTick',[],'YTick',[],'FontName','Courier')
end
text(-2.7,2.45,'h)','FontSize',font,'FontName','Courier')

subplot('Position',[0.63 0.07 0.25 0.27])
pcolor(squeeze(Z_cSPH(:,:,fix(end/2)))/r_SPH,(squeeze(X_cSPH(:,:,fix(end/2)))-rco_SPH)/r_SPH,squeeze(T_RR_SPH(:,:,fix(end/2)))/BUO_SPH)
e=colorbar;
axis image,shading interp,xlabel('\bf{\theta/R}')
if paper == 1
    clim([-0.2,0.2])
    if equal==1
        clim([-1 1])
    end
    set(e,'Position',[0.885 0.075 0.02 0.26],'FontSize',font_ax)
    set(gca,'YTick',[],'FontSize',font_ax,'FontName','Courier')
end
text(-2.7,2.45,'i)','FontSize',font,'FontName','Courier')


set(gcf,'Position',[349 100 940 847])

if Save == 1
    folder = '../docs';
    print([folder 'fig_compare' '.png'],'-dpng','-r300')
end
