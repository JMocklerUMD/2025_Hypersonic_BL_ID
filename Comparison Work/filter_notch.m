function filt_data = filter_notch(data,cut_low,cut_high,order_low,order_high)
  
% NB: _low referes to low-pass filter, _high to high-pass

D = fft(data);
nn = length(D);
ff = 1:nn/2;
if size(D,2) ~= 1
  D = D';
end

% low-pass filter
H1l = sqrt(1./(1+(ff/(cut_low*nn)).^(2*order_low)));
H2l = H1l(end:-1:1);

% Problem if nn oddfil
if mod(nn,2)~=0
  HIl = sqrt(1./(1+(ceil(nn/2)/(cut_low*nn)).^(2*order_low)));
else
  HIl = [];
end

Hl = [H1l,HIl,H2l]';

% high-pass filter
H1h = sqrt(1./(1+(cut_high*nn./ff).^(2*order_high)));
H2h = H1h(end:-1:1);

% Problem if nn odd
if mod(nn,2)~=0
  HIh = sqrt(1./(1+(cut_high*nn/ceil(nn/2)).^(2*order_high)));
else
  HIh = [];
end

Hh = [H1h,HIh,H2h]';

D = D.*Hl.*Hh;

filt_data = real(ifft(D));
