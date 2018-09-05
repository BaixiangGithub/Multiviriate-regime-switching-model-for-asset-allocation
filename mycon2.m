function [c,ceq] = mycon2(alpha)
%[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_y_H, epsilon_pi_H, sigma_yL, sigma_piL, beta1, beta2, beta3, beta4] 
%[u_y_l, gamma_y, u_pi_l, gamma_pi, epsilon_yH, epsilon_piH, sigma_yL, sigma_piL, beta1 ,beta2,  beta3,beta4]
u_y_l = alpha(1); gamma_y = alpha(2); u_pi_l = alpha(3); gamma_pi = alpha(4);epsilon_y_H = alpha(5);
epsilon_pi_H = alpha(6);sigma_yL = alpha(7); sigma_piL = alpha(8);beta1= alpha(9); beta2 = alpha(10); beta3 = alpha(11); beta4 = alpha(12);
%sigma_pi = alpha(5); sigma_y = alpha(6); rho = alpha(7);
%{
A = [-1, 0 , -gamma_pi, 0, 0, 0, 0, 0,0,0,0,0
     0 , -1, 0, -gamma_y, 0, 0, 0, 0,0,0,0,0 
     0 , 0, 0, 0, epsilon_y_H, 0 , -1, 0,0,0,0,0
     0 , 0, 0 , 0 , 0 , epsilon_pi_H, 0 , -1,0,0,0,0
     0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0
     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1
     ];
%}
A = [-1, 0 , -gamma_pi,0 ;
     0 , -1, 0, -gamma_y] ;
c = A* [u_pi_l;u_y_l;gamma_pi;gamma_y];
ceq = zeros(4,4)*[u_pi_l;u_y_l;gamma_pi;gamma_y]; 
%size(A)

%c = A* [u_pi_l;u_y_l;gamma_pi;gamma_y;epsilon_y_H;epsilon_pi_H;sigma_yL;sigma_piL;beta1*(sigma_yL-epsilon_y_H^2)^2;beta2*(sigma_yL-epsilon_y_H^2)^2;...
%    sigma_yL^2*beta3; sigma_yL^2*beta4];
%ceq = zeros(4,8)*[u_pi_l;u_y_l;gamma_pi;gamma_y;epsilon_y_H;epsilon_pi_H;sigma_yL;sigma_piL]; 