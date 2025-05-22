%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               This verification of the first identity between Rosenbluth 
%               potentials is based on the article "Fast convolution 
%               with free-space Green's function" by F. Vico, L. Greengard, and M.
%               Ferrando, Journal of Computational Physics 323 (2016)
%               191-203
%
%               % Copyright (C) 2018-2023: Junyi Zou and Antoine Cerfon
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
N=64;%number of points in x-domain - must be an even number
L=1.8; %Size of truncation window
h=1/N; %x_step
[x,y,z]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h,-1/2:h:1/2-h); % % Computational grid of N regularly spaced points in each dimension

% Define source distribution - take 3d gaussian as an example
sigma=0.05;
r2=x.^2+y.^2+z.^2; r=sqrt(r2); %square of radius
rho=1/((2*pi)^(3/2)*sigma^3)*exp(-r2/(2*sigma^2)); % A 3d gaussian distribution

% Construct corresponding Fourier domain - with higher resolution to
% account for the need for padding and for the oscillatory nature of the
% kernel, as discussed in the article by Vico et al.
hs=pi/2; %s_step
[wm1,wm2,wm3]=ndgrid([0:hs:N*pi-hs -N*pi:hs:-hs],[0:hs:N*pi-hs -N*pi:hs:-hs],[0:hs:N*pi-hs -N*pi:hs:-hs]); %s-domain

% Prepare tools to compute 
sf = pi*[0:N-1, 0, -N+1:-1];% Construct doubly refined Fourier domain for differentiation to get diffusion tensor
[sfx,sfy,sfz]=ndgrid(sf,sf,sf);%Wavevectors in two dimensions

% Define high order mollified Green's functions
s=sqrt(wm1.^2+wm2.^2+wm3.^2); %radius in s-domain
greenBiharmonic=((2-L^2*s.^2).*cos(L*s)+2*L*s.*sin(L*s)-2)./(2*s.^4);%mollified green's function for the biharmonic equation
greenBiharmonic(1,1,1)=L^4/8;

greenPoisson=2*(sin(L*s/2)./s).^2;%mollified green's function for Poisson's equation
greenPoisson(1,1,1)=L^2/2;

%Precomputation for the biharmonic equation
T1Biharmonic=ifftn(greenBiharmonic); %precomputation for T

TBiharmonic = zeros(2*N,2*N,2*N);
TBiharmonic(1:N,1:N,1:N)=T1Biharmonic(1:N,1:N,1:N);

for p = 1:N
    for q = 1:N
        for r = 1:N
        TBiharmonic(2*N-(p-1),q,r)=T1Biharmonic(p+1,q,r);
        TBiharmonic(p,2*N-(q-1),r)=T1Biharmonic(p,q+1,r);
        TBiharmonic(p,q,2*N-(r-1))=T1Biharmonic(p,q,r+1);
        TBiharmonic(2*N-(p-1),2*N-(q-1),2*N-(r-1))=T1Biharmonic(p+1,q+1,r+1);
        TBiharmonic(2*N-(p-1),2*N-(q-1),r)=T1Biharmonic(p+1,q+1,r);
        TBiharmonic(2*N-(p-1),q,2*N-(r-1))=T1Biharmonic(p+1,q,r+1);
        TBiharmonic(p,2*N-(q-1),2*N-(r-1))=T1Biharmonic(p,q+1,r+1);
        end
    end
end


%Precomputation for Poisson's equation
T1Poisson=ifftn(greenPoisson); %precomputation for T

TPoisson = zeros(2*N,2*N,2*N);
TPoisson(1:N,1:N,1:N)=T1Poisson(1:N,1:N,1:N);

for p = 1:N
    for q = 1:N
        for r = 1:N
        TPoisson(2*N-(p-1),q,r)=T1Poisson(p+1,q,r);
        TPoisson(p,2*N-(q-1),r)=T1Poisson(p,q+1,r);
        TPoisson(p,q,2*N-(r-1))=T1Poisson(p,q,r+1);
        TPoisson(2*N-(p-1),2*N-(q-1),2*N-(r-1))=T1Poisson(p+1,q+1,r+1);
        TPoisson(2*N-(p-1),2*N-(q-1),r)=T1Poisson(p+1,q+1,r);
        TPoisson(2*N-(p-1),q,2*N-(r-1))=T1Poisson(p+1,q,r+1);
        TPoisson(p,2*N-(q-1),2*N-(r-1))=T1Poisson(p,q+1,r+1);
        end
    end
end


%Compute the first Rosenbluth potential
pot=real(ifftn(fftn(TPoisson).*fftn(rho,[2*N 2*N 2*N])));%standard aperiod conv with 2N padding
pot=pot(1:N,1:N,1:N);% Rosenbluth potential restricted to computational domain

% Diagonal entries of the diffusion tensor
Dxx=real(ifftn(-sfx.^2.*fftn(TBiharmonic).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dxx=Dxx(1:N,1:N,1:N);% xx-entry of diffusion tensor restricted to computational domain
Dyy=real(ifftn(-sfy.^2.*fftn(TBiharmonic).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dyy=Dyy(1:N,1:N,1:N);% yy-entry of diffusion tensor restricted to computational domain
Dzz=real(ifftn(-sfz.^2.*fftn(TBiharmonic).*fftn(rho,[2*N 2*N 2*N])));%standard aperiodic convolution with 2N padding
Dzz=Dzz(1:N,1:N,1:N);% zz-entry of diffusion tensor restricted to computational domain

Trace = Dxx+Dyy+Dzz; %Trace of the diffusion tensor

% Define error to verify identity
error = abs(Trace - pot);

% plot results
[xplot,yplot]=ndgrid(-1/2:h:1/2-h,-1/2:h:1/2-h);

figure(1)
subplot(1,3,1)
plot3(xplot,yplot,pot(:,:,N/2));
title('Potential');
subplot(1,3,2)
plot3(xplot,yplot,Trace(:,:,N/2));
title('Trace');
subplot(1,3,3)
plot3(xplot,yplot,error(:,:,N/2));
title('Identity Error');