function y = maprange(array,minInit,maxInit)
    y = (array - minInit) * (65535 - 0) / (maxInit - minInit) + 0;
end
%maps to min and max value of uint16