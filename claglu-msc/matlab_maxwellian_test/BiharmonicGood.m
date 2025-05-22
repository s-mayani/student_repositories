%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               This fast, simple and accurate free space biharmonic solver 
%               is based on the article "Fast convolution with free-space
%               Green's function" by F. Vico, L. Greengard, and M.
%               Ferrando, Journal of Computational Physics 323 (2016)
%               191-203
%
%               We use it to compute the diffusion tensor that appears in
%               the Fokker-Planck collision operator
%
%               % Copyright (C) 2023: Antoine Cerfon
%               Contact: cerfon@cims.nyu.edu
% 
%               This program is free software; you can redistribute it and/or modify 
%               it under the terms of the GNU General Public License as published by 
%               the Free Software Foundation; either version 2 of the License, or 
%               (at your option) any later version.  This program is distributed in 
%               the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
%               even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
%               PARTICULAR PURPOSE.  See the GNU General Public License for more 
%               details. You should have received a copy of the GNU General Public 
%               License along with this program; if not, see <http://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%Initialization
N=32;%number of points in x-domain - must be an even number
L=1.8; %Size of truncation window
h=1/N; %x_step
[x,y,z]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h,-1/2:h:1/2-h); % % Computational grid of N regularly spaced points in each dimension

% Define source for biharmonic problem Delta^2 f = -rho
sigma=0.05;
r2=x.^2+y.^2+z.^2; r=sqrt(r2); %square of radius
rho=1/((2*pi)^(3/2)*sigma^3)*exp(-r2/(2*sigma^2)); % A 3d gaussian distribution

% Exact solutions
% Potential
exact_pot=1/(8*pi)*(sigma*sqrt(2/pi)*exp(-r2/(2*sigma^2))+erf(r/(sigma*sqrt(2))).*(sigma^2./r+r)); %Exact biharmonic problem
exact_pot(N/2+1,N/2+1,N/2+1)=sigma/((2*pi)^(3/2)); % Exact value at the apparent singularity at r=0

% Exact Gradient
% x-component
exact_pot_x=1/(8*pi)*(-sqrt(2/pi)*x/sigma.*exp(-r2/(2*sigma^2))...
    +sqrt(2/pi)/sigma*x./r.*(sigma^2./r+r).*exp(-r2/(2*sigma^2))...
    +erf(r/(sigma*sqrt(2))).*(x./r-sigma^2*x./r.^3));

exact_pot_x(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0

exact_pot_y=1/(8*pi)*(-sqrt(2/pi)*y/sigma.*exp(-r2/(2*sigma^2))...
    +sqrt(2/pi)/sigma*y./r.*(sigma^2./r+r).*exp(-r2/(2*sigma^2))...
    +erf(r/(sigma*sqrt(2))).*(y./r-sigma^2*y./r.^3));

exact_pot_y(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0

exact_pot_z=1/(8*pi)*(-sqrt(2/pi)*z/sigma.*exp(-r2/(2*sigma^2))...
    +sqrt(2/pi)/sigma*z./r.*(sigma^2./r+r).*exp(-r2/(2*sigma^2))...
    +erf(r/(sigma*sqrt(2))).*(z./r-sigma^2*z./r.^3));

exact_pot_z(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0

% Exact diffusion tensor D
% The tensor D is symmetric, so we only compute 6 entries
% xx entry
exact_Dxx = 1/(8*pi)*(sqrt(2/pi)*1/sigma*exp(-r2/(2*sigma^2)).*...
    ((x.^2-sigma^2)/sigma^2+(sigma^2/r+r).*((sigma^2*r2...
     -x.^2.*r2-sigma^2*x.^2)./(sigma^2*r.^3))+2*x.^2./r2.^2.*(r2-sigma^2))...
     +1/r.^5.*erf(r/(sigma*sqrt(2))).*(r2.^2-r2.*(x.^2+sigma^2)+3*sigma^2*x.^2));
 
 exact_Dxx(N/2+1,N/2+1,N/2+1)=1/(6*sqrt(2)*sigma*pi^(3/2)); % Exact value at the apparent singularity at r=0
 
 % xy entry
 exact_Dxy = 1/(8*pi)*(-3*sigma*sqrt(2/pi)*x.*y./r2.^2.*exp(-r2/(2*sigma^2))...
             +erf(r/(sigma*sqrt(2))).*(x.*y./r.^3.*(3*sigma^2./r2-1)));
         
 exact_Dxy(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0
 
 % xz entry
 exact_Dxz = 1/(8*pi)*(-3*sigma*sqrt(2/pi)*x.*z./r2.^2.*exp(-r2/(2*sigma^2))...
             +erf(r/(sigma*sqrt(2))).*(x.*z./r.^3.*(3*sigma^2./r2-1)));
         
 exact_Dxz(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0
 
% yy entry
exact_Dyy = 1/(8*pi)*(sqrt(2/pi)*1/sigma*exp(-r2/(2*sigma^2)).*...
    ((y.^2-sigma^2)/sigma^2+(sigma^2/r+r).*((sigma^2*r2...
     -y.^2.*r2-sigma^2*y.^2)./(sigma^2*r.^3))+2*y.^2./r2.^2.*(r2-sigma^2))...
     +1/r.^5.*erf(r/(sigma*sqrt(2))).*(r2.^2-r2.*(y.^2+sigma^2)+3*sigma^2*y.^2));
 
exact_Dyy(N/2+1,N/2+1,N/2+1)=1/(6*sqrt(2)*sigma*pi^(3/2)); % Exact value at the apparent singularity at r=0

 % yz entry
 exact_Dyz = 1/(8*pi)*(-3*sigma*sqrt(2/pi)*y.*z./r2.^2.*exp(-r2/(2*sigma^2))...
             +erf(r/(sigma*sqrt(2))).*(y.*z./r.^3.*(3*sigma^2./r2-1)));
         
 exact_Dyz(N/2+1,N/2+1,N/2+1)=0; % Exact value at the apparent singularity at r=0

% zz entry
exact_Dzz = 1/(8*pi)*(sqrt(2/pi)*1/sigma*exp(-r2/(2*sigma^2)).*...
    ((z.^2-sigma^2)/sigma^2+(sigma^2/r+r).*((sigma^2*r2...
     -z.^2.*r2-sigma^2*z.^2)./(sigma^2*r.^3))+2*z.^2./r2.^2.*(r2-sigma^2))...
     +1/r.^5.*erf(r/(sigma*sqrt(2))).*(r2.^2-r2.*(z.^2+sigma^2)+3*sigma^2*z.^2));
 
exact_Dzz(N/2+1,N/2+1,N/2+1)=1/(6*sqrt(2)*sigma*pi^(3/2)); % Exact value at the apparent singularity at r=0
 

% Construct corresponding Fourier domain - with higher resolution to
% account for the need for padding and for the oscillatory nature of the
% kernel, as discussed in the article by Vico et al.
hs=pi/2; %s_step
[wm1,wm2,wm3]=ndgrid(-N*pi:hs:N*pi-hs,-N*pi:hs:N*pi-hs,-N*pi:hs:N*pi-hs); %s-domain

% Compute electric field
sf = pi*[0:N-1, 0, -N+1:-1];% Construct doubly refined Fourier domain for differentiation to get electric field
[sfx,sfy,sfz]=ndgrid(sf,sf,sf);%Wavevectors in three dimensions

% Define high order mollified Green's function
s=sqrt(wm1.^2+wm2.^2+wm3.^2); %radius in s-domain
green=((2-L^2*s.^2).*cos(L*s)+2*L*s.*sin(L*s)-2)./(2*s.^4);%mollified green's function;%modified green's function
green(2*N+1,2*N+1,2*N+1)=L^4/8;

% Construct extended domain required for padding
[xx,yy,zz]=ndgrid(-2:h:2-h,-2:h:2-h,-2:h:2-h); %extended x-domain 
constant=exp(-1i*xx*N*pi).*exp(-1i*yy*N*pi).*exp(-1i*zz*N*pi);%precomputation for T - this constant is required to turn the desired integral with the trapezoidal rule into a form which can be computed directly with MATLAB's FFT

%Precomputation
T1=ifftn(green).*constant; %precomputation for T
T(:,:,1:N) = [T1(1:N,1:N,1:N) T1(1:N,3*N+1:4*N,1:N); T1(3*N+1:4*N,1:N,1:N) T1(3*N+1:4*N,3*N+1:4*N,1:N)]; 
T(:,:,N+1:2*N)=[T1(1:N,1:N,3*N+1:4*N) T1(1:N,3*N+1:4*N,3*N+1:4*N); T1(3*N+1:4*N,1:N,3*N+1:4*N) T1(3*N+1:4*N,3*N+1:4*N,3*N+1:4*N)];

% Compute free space solution
%Potential
result=real(ifftn(fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiod conv with 2N padding
pot=result(1:N,1:N,1:N); %Potential restricted to computational domain

% Gradient
pot_x=real(ifftn(1i*sfx.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 
pot_y=real(ifftn(1i*sfy.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 
pot_z=real(ifftn(1i*sfz.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding 

pot_x=pot_x(1:N,1:N,1:N);% x-component of gradient restricted to computational domain
pot_y=pot_y(1:N,1:N,1:N);% y-component of gradient restricted to computational domain
pot_z=pot_z(1:N,1:N,1:N);% z-component of gradient restricted to computational domain

% Diffusion tensor
Dxx=real(ifftn(-sfx.^2.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dxx=Dxx(1:N,1:N,1:N);% xx-entry of diffusion tensor restricted to computational domain
Dxy=real(ifftn(-sfx.*sfy.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dxy=Dxy(1:N,1:N,1:N);% xy-entry of diffusion tensor restricted to computational domain
Dxz=real(ifftn(-sfx.*sfz.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dxz=Dxz(1:N,1:N,1:N);% xy-entry of diffusion tensor restricted to computational domain
Dyy=real(ifftn(-sfy.^2.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dyy=Dyy(1:N,1:N,1:N);% yy-entry of diffusion tensor restricted to computational domain
Dyz=real(ifftn(-sfy.*sfz.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dyz=Dyz(1:N,1:N,1:N);% xy-entry of diffusion tensor restricted to computational domain
Dzz=real(ifftn(-sfz.^2.*fftn(T).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dzz=Dzz(1:N,1:N,1:N);% zz-entry of diffusion tensor restricted to computational domain


% Compute errors
relative_errorpot=abs((pot-exact_pot)./exact_pot);% Potential

error_pot_x = abs(pot_x-exact_pot_x); % x-component of gradient
error_pot_y = abs(pot_y-exact_pot_y); % y-component of gradient
error_pot_z = abs(pot_z-exact_pot_z); % z-component of gradient

error_Dxx = abs(Dxx-exact_Dxx); % xx-entry of diffusion tensor
error_Dxy = abs(Dxy-exact_Dxy); % xy-entry of diffusion tensor
error_Dxz = abs(Dxz-exact_Dxz); % xz-entry of diffusion tensor
error_Dyy = abs(Dyy-exact_Dyy); % yy-entry of diffusion tensor
error_Dyz = abs(Dyz-exact_Dyz); % yz-entry of diffusion tensor
error_Dzz = abs(Dzz-exact_Dzz); % zz-entry of diffusion tensor

% Output errors
pot_err_disp = ['The maximum relative error for the potential is ', num2str(max(max(max(relative_errorpot))))];
pot_x_err_disp = ['The maximum error for the x coordinate of the gradient is ', num2str(max(max(max(error_pot_x))))];
pot_y_err_disp = ['The maximum error for the y coordinate of the gradient is ', num2str(max(max(max(error_pot_y))))];
pot_z_err_disp = ['The maximum error for the z coordinate of the gradient is ', num2str(max(max(max(error_pot_z))))];
Dxx_err_disp = ['The maximum error for the xx entry of the diffusion tensor is ', num2str(max(max(max(error_Dxx))))];
Dxy_err_disp = ['The maximum error for the xy entry of the diffusion tensor is ', num2str(max(max(max(error_Dxy))))];
Dxz_err_disp = ['The maximum error for the xz entry of the diffusion tensor is ', num2str(max(max(max(error_Dxz))))];
Dyy_err_disp = ['The maximum error for the yy entry of the diffusion tensor is ', num2str(max(max(max(error_Dyy))))];
Dyz_err_disp = ['The maximum error for the yz entry of the diffusion tensor is ', num2str(max(max(max(error_Dyz))))];
Dzz_err_disp = ['The maximum error for the zz entry of the diffusion tensor is ', num2str(max(max(max(error_Dzz))))];

% Display errors in Command Window
disp(pot_err_disp);
disp(pot_x_err_disp);
disp(pot_y_err_disp);
disp(pot_z_err_disp);
disp(Dxx_err_disp);
disp(Dxy_err_disp);
disp(Dxz_err_disp);
disp(Dyy_err_disp);
disp(Dyz_err_disp);
disp(Dzz_err_disp);

% plot results for illustrative purposes
[xplot,yplot]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h);


figure(1)
subplot(1,3,1)
plot3(xplot,yplot,pot(:,:,N/2));
title('Potential Numerical');
subplot(1,3,2)
plot3(xplot,yplot,exact_pot(:,:,N/2));
title('Potential Exact');
subplot(1,3,3)
plot3(xplot,yplot,relative_errorpot(:,:,N/2));
title('Potential Relative Error');

figure(2)
subplot(3,3,1)
plot3(x(:,:,N/2),y(:,:,N/2),pot_x(:,:,N/2));
title('pot_x Numerical');
subplot(3,3,2)
plot3(x(:,:,N/2),y(:,:,N/2),exact_pot_x(:,:,N/2));
title('pot_x Exact');
subplot(3,3,3)
plot3(x(:,:,N/2),y(:,:,N/2),error_pot_x(:,:,N/2));
title('pot_x Absolute Error');
subplot(3,3,4)
plot3(x(:,:,N/2),y(:,:,N/2),pot_y(:,:,N/2));
title('pot_y Numerical');
subplot(3,3,5)
plot3(x(:,:,N/2),y(:,:,N/2),exact_pot_y(:,:,N/2));
title('pot_y Exact');
subplot(3,3,6)
plot3(x(:,:,N/2),y(:,:,N/2),error_pot_y(:,:,N/2));
title('pot_y Absolute Error');
subplot(3,3,7)
plot3(x(:,:,N/2),y(:,:,N/2),pot_z(:,:,N/2));
title('pot_z Numerical');
subplot(3,3,8)
plot3(x(:,:,N/2),y(:,:,N/2),exact_pot_z(:,:,N/2));
title('pot_z Exact');
subplot(3,3,9)
plot3(x(:,:,N/2),y(:,:,N/2),error_pot_z(:,:,N/2));
title('pot_z Absolute Error');

figure(3)
subplot(6,3,1)
plot3(x(:,:,N/2),y(:,:,N/2),Dxx(:,:,N/2));
title('Dxx Numerical');
subplot(6,3,2)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dxx(:,:,N/2));
title('Dxx Exact');
subplot(6,3,3)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dxx(:,:,N/2));
title('Dxx Absolute Error');
subplot(6,3,4)
plot3(x(:,:,N/2),y(:,:,N/2),Dxy(:,:,N/2));
title('Dxy Numerical');
subplot(6,3,5)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dxy(:,:,N/2));
title('Dxy Exact');
subplot(6,3,6)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dxy(:,:,N/2));
title('Dxy Absolute Error');
subplot(6,3,7)
plot3(x(:,:,N/2),y(:,:,N/2),Dxz(:,:,N/2));
title('Dxz Numerical');
subplot(6,3,8)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dxz(:,:,N/2));
title('Dxz Exact');
subplot(6,3,9)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dxz(:,:,N/2));
title('Dxz Absolute Error');
subplot(6,3,10)
plot3(x(:,:,N/2),y(:,:,N/2),Dyy(:,:,N/2));
title('Dyy Numerical');
subplot(6,3,11)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dyy(:,:,N/2));
title('Dyy Exact');
subplot(6,3,12)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dyy(:,:,N/2));
title('Dyy Absolute Error');
subplot(6,3,13)
plot3(x(:,:,N/2),y(:,:,N/2),Dyz(:,:,N/2));
title('Dyz Numerical');
subplot(6,3,14)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dyz(:,:,N/2));
title('Dyz Exact');
subplot(6,3,15)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dyz(:,:,N/2));
title('Dyz Absolute Error');
subplot(6,3,16)
plot3(x(:,:,N/2),y(:,:,N/2),Dzz(:,:,N/2));
title('Dzz Numerical');
subplot(6,3,17)
plot3(x(:,:,N/2),y(:,:,N/2),exact_Dzz(:,:,N/2));
title('Dzz Exact');
subplot(6,3,18)
plot3(x(:,:,N/2),y(:,:,N/2),error_Dzz(:,:,N/2));
title('Dzz Absolute Error');
