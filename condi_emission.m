function [emission_density] = condi_emission(u_y_l, gamma_y, u_pi_l, gamma_pi, sigma_y,sigma_pi, beta,yt,Pit,s_t_y,s_t_pi)




MU = [u_y_l + s_t_y*gamma_y^2,u_pi_l+s_t_pi*gamma_pi^2];
SIGMA = [sigma_y^2, sigma_y^2*beta; sigma_y^2*beta, sigma_y^2 * beta^2 + sigma_pi^2];

emission_density = mvnpdf([yt,Pit],MU,SIGMA);





%emission_density =exp(- [yt-u_y_l - s_t_y*gamma_y^2;Pit-u_pi_l-s_t_pi*gamma_pi^2]'*([SIGMA(2,2),-SIGMA(1,2);-SIGMA(2,1),SIGMA(1,1)]./(sigma_y*sigma_pi)^2)*[yt-u_y_l - s_t_y*gamma_y^2;Pit-u_pi_l-s_t_pi*gamma_pi^2]/2 );


