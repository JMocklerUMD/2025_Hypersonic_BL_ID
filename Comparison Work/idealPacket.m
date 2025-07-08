function [packet] = idealPacket(BL_thickness,centerWidth,x,center,frequencyFactor)
%% [packet] = idealPacket(BL_thickness,centerWidth,x,center,frequencyFactor)
% 
%   Inputs:
%       BL_thickness - the thickness of the boundary layer
%       centerWidth - the width of the section that has an amplitude of 1
%       x - the location vector
%       center - the x value corrosponding to the center of the packet
%       frequencyFactor - the number of boundary layer thicknesses that
%           corrospond to the wavelength of the packet.
%
%   Output:
%       packet - the expected values of the wave packet at corrosponding to
%           x
%

omega = 2*pi/(frequencyFactor*BL_thickness);
sigma = 2*BL_thickness;

packet = zeros(size(x));

x = x-center;

for i = 1:length(x)
    if x(i)<-centerWidth/2
        packet(i) = exp(-(x(i)+centerWidth/2).^2./(2*sigma.^2)).*exp(1i*omega*(x(i)));
    elseif x(i)>centerWidth/2
        packet(i) = exp(-(x(i)-centerWidth/2).^2./(2*sigma.^2)).*exp(1i*omega*(x(i)));
    else
        packet(i) = exp(1i*omega*x(i));
    end   
end


end